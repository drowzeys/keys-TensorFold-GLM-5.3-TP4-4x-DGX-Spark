#!/usr/bin/env python3
"""DFlash2 drafter fine-tune for GLM-5.3 against the served EXL3 target (on-policy data captured from TensorFold).

Standalone training-time mirror of the fork's inference path (image glm53-exl3-tp4:keys19+):
  vllm/model_executor/models/qwen3_dflash.py  (DFlashQwen3Model: fc -> hidden_norm -> fused ctx K/V,
                                               k_norm + RoPE, block queries = [anchor, mask x K])
  vllm/model_executor/models/qwen3_dflash2.py (grouped 2-tap dynamic convs, CandidateSelector/_score_edges)
  vllm/v1/spec_decode/dflash.py + utils.copy_and_expand_dflash_inputs_kernel
      -> context = every token whose target hidden state exists (positions <= anchor-1),
         query   = [anchor token (bonus), mask x K] at positions anchor..anchor+K,
         sampled = the K MASK slots only (query_off > 0); the anchor slot's logits are never used.
  vllm/model_executor/models/deepseek_v2.py -> aux_hidden_states[i] = residual stream ENTERING target layer
         (target_layer_ids[i] + 1), i.e. the output of layer target_layer_ids[i]; concatenated in list order.

Loss = sum_j w_j * CE(logits[mask slot j], ids[anchor+j]), j=1..K, w_j = exp(-(j-1)/GAMMA)   (DFlash paper, gamma=4 @ block 8)
     + SEL_W * selector path loss: at step s (mask slot s+1) the top-k candidates of the drafter logits are scored with
       S[s,p,c] = U[s,c] + <A(pred_p) * H(h_s), B(cand_c)> (exactly qwen3_dflash2._score_edges); CE over c of S[s, p*, :]
       where p* = index of the TRUE token at step s-1 among step s-1 candidates (0 at s=0: all rows are the anchor).
       Steps where the true chain leaves the top-k are masked out. This is the calibration the 27B retrain lacked.

TRAIN_SET: full  = everything except embed/lm_head (2.46B; ~46 GB w/ fp32 master + AdamW; peak ~55 GB on a DGX Spark)
           lora  = fc + selector + norms + convs full, LoRA(r=LORA_R) on q/k/v/o/gate/up/down (~0.46B trainable; ~18 GB)
           light = fc + selector + norms + convs only (~0.33B; ~14 GB)
Data: SEQ_DIR/seq-*.safetensors {ids int32 [T], pos int32 [T], aux bf16 [T, 6*H]} written by data/assemble.py.
Modes: MODE=train | parity (no update; report top-1 agreement per slot + selector chain accuracy) | synth (tiny model, toy stream, no files) | roundtrip (load real ckpt, export, assert layout)
Output: OUT_DIR/model.safetensors in the incoai key layout (asserted identical key set + shapes to DRAFT_DIR) + config.json copy.
"""
import glob, json, math, os, sys, time, random, threading, queue
import torch, torch.nn as nn, torch.nn.functional as F

E = os.environ.get
MODE = E("MODE", "train")
SYNTH = MODE == "synth"
DRAFT = E("DRAFT_DIR", "/models/GLM-5.3-DFlash2")
TARGET = E("TARGET_DIR", "/models/GLM-5.3-EXL3-2.75BPW")
SEQ_DIR = E("SEQ_DIR", "/data/seq")
OUT = E("OUT_DIR", os.path.expanduser("~/dflash2-glm53-ft/runs/ft1"))
DEV = E("DEV", "cuda" if torch.cuda.is_available() else "cpu")
TRAIN_SET = E("TRAIN_SET", "lora")
LORA_R = int(E("LORA_R", "64")); LORA_ALPHA = float(E("LORA_ALPHA", "128"))
STEPS = int(E("STEPS", "2000")); LR = float(E("LR", "1e-4" if TRAIN_SET != "full" else "2e-5"))
WARMUP = float(E("WARMUP", "0.04")); WD = float(E("WD", "0.0"))
B = int(E("BATCH", "2")); T = int(E("CTX", "2048")); G = int(E("ANCHORS", "64"))
GAMMA = float(E("GAMMA", "4")); SEL_W = float(E("SEL_W", "1.0")); SEL_DETACH_H = E("SEL_DETACH_H", "1") == "1"
LOSS = E("LOSS", "ce")                 # ce | kl (soft top-K target) | hybrid (HYB_ALPHA*kl + (1-HYB_ALPHA)*ce); kl/hybrid fall back to ce on rows without captured top-K
TAIL = E("TAIL", "bucket")             # bucket: exact tail mass 1-sum(exp(l_k-lse)) as a (K+1)-th class | renorm: renormalize over the K candidates
SOFT_TEMP = float(E("SOFT_TEMP", "1.0"))   # temperature applied to the target top-K logits (1.0 = match raw logits; the serve applies the request temperature to both sides)
HYB_ALPHA = float(E("HYB_ALPHA", "0.5"))
POS_WEIGHT = E("POS_WEIGHT", "fixed")  # fixed: exp(-(j-1)/GAMMA) | dpace: acceptance-conditioned (EMA of P(greedy chain reaches slot j)), shared with the selector loss
SEL_POSW = E("SEL_POSW", "0") == "1"   # selector loss uses the same position weights as the block loss
SEL_TARGET = E("SEL_TARGET", "hard")   # hard: true token index | soft: target top-K mass restricted to the draft's candidates (falls back to hard)
MEM_FRACTION = float(E("MEM_FRACTION", "0"))   # cap the CUDA caching allocator (GB10 unified memory: an overcommit swaps the node)
SEED = int(E("SEED", "0")); SAVE_EVERY = int(E("SAVE_EVERY", "500")); LOG_EVERY = int(E("LOG_EVERY", "10"))
HOLDOUT_FRAC = float(E("HOLDOUT_FRAC", "0.02")); SKIP_SHARED = E("SKIP_SHARED", "0") == "1"
torch.manual_seed(SEED); random.seed(SEED)
if DEV.startswith("cuda") and MEM_FRACTION > 0:
    torch.cuda.set_per_process_memory_fraction(MEM_FRACTION)

# ---------------- config ----------------
if SYNTH:
    cfg = dict(hidden_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=64, intermediate_size=512,
               num_hidden_layers=2, vocab_size=2048, rms_norm_eps=1e-5, rope_parameters={"rope_theta": 1e6},
               sliding_window=2048, dflash_config=dict(block_size=8, conv_group_size=16, conv_kernel_size=2,
               mask_token_id=2047, selector_rank=32, selector_top_k=8, target_layer_ids=[1, 3, 5, 7, 9, 11]))
else:
    cfg = json.load(open(f"{DRAFT}/config.json"))
dc = cfg["dflash_config"]
HID = cfg["hidden_size"]; NH = cfg["num_attention_heads"]; NKV = cfg["num_key_value_heads"]; HD = cfg["head_dim"]
INTER = cfg["intermediate_size"]; NLAY = cfg["num_hidden_layers"]; VOCAB = cfg["vocab_size"]; EPS = cfg["rms_norm_eps"]
THETA = float(cfg["rope_parameters"]["rope_theta"]); TAPS = len(dc["target_layer_ids"]); AUXW = HID * TAPS
MASK_ID = int(dc["mask_token_id"]); CK = int(dc["conv_kernel_size"]); CG = int(dc["conv_group_size"]); NGROUP = HID // CG
BLK = int(dc["block_size"]); K = BLK - 1; SEL_R = int(dc["selector_rank"]); SEL_K = int(dc["selector_top_k"])
SW = int(cfg.get("sliding_window") or 0) or 10**9
EMB_SCALE = float(dc.get("input_embedding_scale", 1.0))
assert cfg.get("is_causal", False) is False or SYNTH, "mirror assumes non-causal block attention (incoai: is_causal=false)"
print(f"[cfg] LOSS={LOSS} TAIL={TAIL} SOFT_TEMP={SOFT_TEMP} POS_WEIGHT={POS_WEIGHT} SEL_TARGET={SEL_TARGET} HID={HID} NH={NH} NKV={NKV} HD={HD} L={NLAY} INTER={INTER} V={VOCAB} AUXW={AUXW} BLK={BLK} K={K} "
      f"sel(r={SEL_R},k={SEL_K}) conv(k={CK},g={CG}) SW={SW} mask={MASK_ID} set={TRAIN_SET} dev={DEV}", flush=True)

# ---------------- model mirror ----------------
class RMSNorm(nn.Module):
    def __init__(s, n): super().__init__(); s.weight = nn.Parameter(torch.ones(n))
    def forward(s, x):
        xf = x.float(); xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
        return (xf.to(x.dtype) * s.weight.to(x.dtype))

class Rope:
    def __init__(s, dim, theta, maxp):
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        f = torch.outer(torch.arange(maxp).float(), inv); s.cos = f.cos().to(DEV); s.sin = f.sin().to(DEV)
    def __call__(s, x, pos):  # x [..., n, HD], pos [...] ; neox rotate-half (vLLM is_neox=True)
        c = s.cos[pos].unsqueeze(-2).to(x.dtype); sn = s.sin[pos].unsqueeze(-2).to(x.dtype)
        d = x.shape[-1] // 2; x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * c - x2 * sn, x2 * c + x1 * sn], -1)
ROPE = None

def grouped_conv(hs, delta, base):  # mirror of qwen3_dflash2._grouped_conv; hs [N,HID] flat, block position = index & (BLK-1)
    blocks = hs.unflatten(-1, (NGROUP, CG)); coef = base.view(1, CK, NGROUP, CG).to(hs.dtype) + delta.unsqueeze(-1).to(hs.dtype)
    out = coef[:, 0] * blocks
    pos = torch.arange(hs.shape[0], device=hs.device) & (BLK - 1)
    for tap in range(1, CK):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        out = out + coef[:, tap] * shifted * (pos >= tap).view(-1, 1, 1).to(hs.dtype)
    return out.flatten(-2)

class LoRALinear(nn.Module):
    def __init__(s, base: nn.Linear, r, alpha):
        super().__init__(); s.base = base; s.r = r; s.scale = alpha / r
        s.lora_A = nn.Parameter(torch.zeros(r, base.in_features)); s.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(s.lora_A, a=math.sqrt(5))
    def forward(s, x):
        return s.base(x) + (F.linear(F.linear(x, s.lora_A.to(x.dtype)), s.lora_B.to(x.dtype)) * s.scale)
    def merge_(s):
        with torch.no_grad():
            s.base.weight.add_((s.lora_B.float() @ s.lora_A.float() * s.scale).to(s.base.weight.dtype))

class GroupedConv(nn.Module):
    def __init__(s):
        super().__init__()
        s.base_kernel = nn.Parameter(torch.zeros(2, CK, HID)); s.kernel_projection = nn.Linear(HID, 2 * CK * NGROUP, bias=False)
    def prepare(s, hs):
        coef = s.kernel_projection(hs).reshape(hs.shape[0], 2, CK, NGROUP)
        return grouped_conv(hs, coef[:, 0], s.base_kernel[0]), coef[:, 1]
    def finish(s, hs, coef): return grouped_conv(hs, coef, s.base_kernel[1])

class Attn(nn.Module):
    def __init__(s):
        super().__init__()
        s.q_proj = nn.Linear(HID, NH * HD, bias=False); s.k_proj = nn.Linear(HID, NKV * HD, bias=False)
        s.v_proj = nn.Linear(HID, NKV * HD, bias=False); s.o_proj = nn.Linear(NH * HD, HID, bias=False)
        s.q_norm = RMSNorm(HD); s.k_norm = RMSNorm(HD)

class MLP(nn.Module):
    def __init__(s):
        super().__init__()
        s.gate_proj = nn.Linear(HID, INTER, bias=False); s.up_proj = nn.Linear(HID, INTER, bias=False); s.down_proj = nn.Linear(INTER, HID, bias=False)
    def forward(s, x): return s.down_proj(F.silu(s.gate_proj(x)) * s.up_proj(x))

class Layer(nn.Module):
    def __init__(s):
        super().__init__()
        s.input_layernorm = RMSNorm(HID); s.post_attention_layernorm = RMSNorm(HID)
        s.self_attn = Attn(); s.mlp = MLP(); s.attention_conv = GroupedConv(); s.mlp_conv = GroupedConv()
    def ctx_kv(s, x_ctx, pos_ctx):  # context K/V straight from hidden_norm(fc(aux)) — no input_layernorm (fused path in fork)
        a = s.self_attn
        k = a.k_proj(x_ctx).view(*x_ctx.shape[:-1], NKV, HD); k = ROPE(a.k_norm(k), pos_ctx)
        v = a.v_proj(x_ctx).view(*x_ctx.shape[:-1], NKV, HD)
        return k, v
    def forward(s, h, residual, ck, cv, pos_q, mask):
        # h [Bq, Q, HID] block queries (Bq = B, Q = G*BLK flattened request-major so conv pos = idx & 7)
        Bq, Q = h.shape[:2]
        if residual is None: residual = h; hn = s.input_layernorm(h)
        else: residual = residual + h; hn = s.input_layernorm(residual)
        x, coef = s.attention_conv.prepare(hn.reshape(Bq * Q, HID)); x = x.view(Bq, Q, HID)
        a = s.self_attn
        q = ROPE(a.q_norm(a.q_proj(x).view(Bq, Q, NH, HD)), pos_q)
        k = ROPE(a.k_norm(a.k_proj(x).view(Bq, Q, NKV, HD)), pos_q)
        v = a.v_proj(x).view(Bq, Q, NKV, HD)
        keys = torch.cat([ck, k], 1); vals = torch.cat([cv, v], 1)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), keys.transpose(1, 2), vals.transpose(1, 2), attn_mask=mask, enable_gqa=True)
        o = a.o_proj(o.transpose(1, 2).reshape(Bq, Q, NH * HD))
        o = s.attention_conv.finish(o.reshape(Bq * Q, HID), coef).view(Bq, Q, HID)
        residual = residual + o; hn = s.post_attention_layernorm(residual)
        x, coef = s.mlp_conv.prepare(hn.reshape(Bq * Q, HID))
        m = s.mlp(x.view(Bq, Q, HID))
        m = s.mlp_conv.finish(m.reshape(Bq * Q, HID), coef).view(Bq, Q, HID)
        return m, residual

class Selector(nn.Module):
    def __init__(s):
        super().__init__()
        s.hidden_projection = nn.Linear(HID, SEL_R, bias=False)
        s.predecessor_codebook = nn.Parameter(torch.zeros(VOCAB, SEL_R)); s.successor_codebook = nn.Parameter(torch.zeros(VOCAB, SEL_R))
    def scores(s, cand, unary, h, anchor):  # mirror of qwen3_dflash2._score_edges: cand/unary [N,K,k], h [N,K,HID], anchor [N]
        hp = s.hidden_projection(h)                                   # [N,K,r]
        succ = s.successor_codebook[cand]                             # [N,K,k,r]
        pred_ids = torch.cat([anchor[:, None, None].expand(-1, 1, SEL_K), cand[:, :-1]], 1)
        pred = s.predecessor_codebook[pred_ids]                       # [N,K,k,r]
        return unary[:, :, None] + torch.einsum("blpr,blcr->blpc", (pred * hp[:, :, None]).to(succ.dtype), succ)

class Draft(nn.Module):
    def __init__(s):
        super().__init__()
        s.embed_tokens = nn.Embedding(VOCAB, HID); s.lm_head = nn.Linear(HID, VOCAB, bias=False)   # shared from target, frozen
        s.fc = nn.Linear(AUXW, HID, bias=False); s.hidden_norm = RMSNorm(HID); s.norm = RMSNorm(HID)
        s.layers = nn.ModuleList(Layer() for _ in range(NLAY)); s.candidate_selector = Selector()
    def forward(s, aux, pos_ctx, blk_ids, pos_q, mask):
        """aux [B,T,AUXW]; pos_ctx [B,T]; blk_ids [B,G*BLK]; pos_q [B,G*BLK]; mask [B,1,G*BLK,T+G*BLK] bool(True=attend)"""
        x_ctx = s.hidden_norm(s.fc(aux))
        h = s.embed_tokens(blk_ids) * EMB_SCALE; residual = None
        for L in s.layers:
            ck, cv = L.ctx_kv(x_ctx, pos_ctx)
            h, residual = L(h, residual, ck, cv, pos_q, mask)
        hf = s.norm(residual + h)
        return hf   # [B, G*BLK, HID]; logits = lm_head(hf)

# ---------------- weights ----------------
def st_open(p):
    from safetensors import safe_open
    return safe_open(p, framework="pt")

def orig_layout():
    f = st_open(glob.glob(f"{DRAFT}/*.safetensors")[0])
    return {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}, f

def load_target_shared(m):
    """embed_tokens (bf16, shard 1) + lm_head (f16, shard 39) straight from the SERVED checkpoint = what vLLM shares with the draft."""
    idx = json.load(open(f"{TARGET}/model.safetensors.index.json"))["weight_map"]
    for key, dst in (("model.embed_tokens.weight", m.embed_tokens.weight), ("lm_head.weight", m.lm_head.weight)):
        f = st_open(f"{TARGET}/{idx[key]}"); t = f.get_tensor(key)
        assert tuple(t.shape) == tuple(dst.shape), (key, t.shape, dst.shape)
        with torch.no_grad(): dst.copy_(t.to(dst.dtype))
        print(f"[load] {key} <- {idx[key]} {t.dtype} {tuple(t.shape)}", flush=True)

def build():
    global ROPE
    ROPE = Rope(HD, THETA, 1 << 16)
    m = Draft()
    if SYNTH:
        for p in m.parameters(): nn.init.normal_(p, std=0.02)
        with torch.no_grad():
            for L in m.layers:
                L.attention_conv.base_kernel.zero_(); L.attention_conv.base_kernel[:, 0].fill_(1.0)
                L.mlp_conv.base_kernel.zero_(); L.mlp_conv.base_kernel[:, 0].fill_(1.0)
    else:
        shapes, f = orig_layout(); sd = m.state_dict(); n = 0
        for k in shapes:
            assert k in sd, f"checkpoint key {k} has no mirror parameter"
            t = f.get_tensor(k); assert tuple(t.shape) == tuple(sd[k].shape), (k, t.shape, sd[k].shape)
            with torch.no_grad(): sd[k].copy_(t.to(sd[k].dtype)); n += 1
        own = {k for k in sd if not (k.startswith("embed_tokens") or k.startswith("lm_head"))}
        assert own == set(shapes), f"mirror/checkpoint key mismatch: {sorted(own ^ set(shapes))[:10]}"
        print(f"[load] {n} drafter tensors from {DRAFT} (key set identical to checkpoint)", flush=True)
        if not SKIP_SHARED: load_target_shared(m)
    return m

def apply_train_set(m):
    for p in m.parameters(): p.requires_grad_(False)
    def on(mod):
        for p in mod.parameters(): p.requires_grad_(True)
    if TRAIN_SET == "none":
        pass
    elif TRAIN_SET == "full":
        for n, p in m.named_parameters():
            if not (n.startswith("embed_tokens") or n.startswith("lm_head")): p.requires_grad_(True)
    else:
        on(m.fc); on(m.hidden_norm); on(m.norm); on(m.candidate_selector)
        for L in m.layers:
            on(L.input_layernorm); on(L.post_attention_layernorm); on(L.attention_conv); on(L.mlp_conv)
            on(L.self_attn.q_norm); on(L.self_attn.k_norm)
            if TRAIN_SET == "lora":
                a, ml = L.self_attn, L.mlp
                for holder, name in ((a, "q_proj"), (a, "k_proj"), (a, "v_proj"), (a, "o_proj"), (ml, "gate_proj"), (ml, "up_proj"), (ml, "down_proj")):
                    setattr(holder, name, LoRALinear(getattr(holder, name), LORA_R, LORA_ALPHA))
    # trainable params live in fp32 (master); frozen ones in bf16 (autocast does the bf16 matmuls)
    for p in m.parameters(): p.data = p.data.to(torch.float32 if p.requires_grad else torch.bfloat16)
    ntr = sum(p.numel() for p in m.parameters() if p.requires_grad); ntot = sum(p.numel() for p in m.parameters())
    print(f"[train] trainable {ntr/1e6:.1f}M / total {ntot/1e6:.1f}M (incl. shared embed+lm_head)", flush=True)
    return m

def export_state(m):
    """incoai layout: drop shared embed/lm_head, merge LoRA, bf16, key set + shapes asserted against the original."""
    for mod in list(m.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, LoRALinear): child.merge_(); setattr(mod, name, child.base)
    sd = {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in m.state_dict().items()
          if not (k.startswith("embed_tokens") or k.startswith("lm_head"))}
    if not SYNTH:
        shapes, _ = orig_layout()
        assert set(sd) == set(shapes), f"export key mismatch {sorted(set(sd) ^ set(shapes))[:10]}"
        for k in sd: assert tuple(sd[k].shape) == shapes[k], (k, sd[k].shape, shapes[k])
    return sd

def save(m, path):
    from safetensors.torch import save_file
    os.makedirs(path, exist_ok=True); sd = export_state(m)
    save_file(sd, f"{path}/model.safetensors", metadata={"format": "pt"})
    if not SYNTH:
        import shutil; shutil.copy(f"{DRAFT}/config.json", f"{path}/config.json")
    print(f"[save] {path}/model.safetensors ({len(sd)} tensors, {os.path.getsize(path + '/model.safetensors') >> 20} MB)", flush=True)
    return sd

# ---------------- data ----------------
def list_seqs():
    fs = sorted(glob.glob(f"{SEQ_DIR}/seq-*.safetensors"))
    rng = random.Random(1234); rng.shuffle(fs); nh = max(1, int(len(fs) * HOLDOUT_FRAC)) if fs else 0
    return fs[nh:], fs[:nh]

def load_seq(fn):
    f = st_open(fn); ids = f.get_tensor("ids").long(); pos = f.get_tensor("pos").long()
    soft = (f.get_tensor("topk_ids").long(), f.get_tensor("topk_logits").float(), f.get_tensor("topk_lse").float()) if "topk_ids" in f.keys() else None
    if "aux" in f.keys(): aux = f.get_tensor("aux")
    else:  # fp8 capture: aux_fp8 [T, AUXW] e4m3 + aux_scale [T, TAPS] fp32 (per-row, per-tap absmax/448)
        q = f.get_tensor("aux_fp8").view(-1, TAPS, HID).float(); sc = f.get_tensor("aux_scale").float()
        aux = (q * sc[:, :, None]).reshape(-1, AUXW).to(torch.bfloat16)
    return ids, pos, aux, soft

_SYNTH_TABLE = None
def synth_seq(rng):
    """Learnable toy stream: ids follow a fixed affine rule, aux = table[id] + noise, so CE/top-1 must move within ~50 steps."""
    global _SYNTH_TABLE
    if _SYNTH_TABLE is None:
        _SYNTH_TABLE = torch.randn(VOCAB, AUXW, generator=torch.Generator().manual_seed(1))
    Tn = rng.randint(64, 256); ids = (rng.randint(0, 63) + torch.arange(Tn)) % 64      # cyclic: slot j must learn anchor+j
    # synthetic soft labels: target puts ~0.7 on the true next token, rest on 3 random ids; K=4; lse consistent with a tail of ~5%
    Kt = 4; nxt = torch.roll(ids, -1); tk_ids = torch.stack([nxt] + [torch.randint(0, VOCAB - 1, (Tn,)) for _ in range(Kt - 1)], 1)
    tk_log = torch.tensor([math.log(0.70), math.log(0.15), math.log(0.07), math.log(0.03)]).expand(Tn, Kt).clone(); tk_lse = torch.zeros(Tn)
    tk_ids[-1] = -1                                  # last row: no logits (prompt-like) -> exercises the CE fallback
    return ids, torch.arange(Tn), (_SYNTH_TABLE[ids] + 0.1 * torch.randn(Tn, AUXW)).to(torch.bfloat16), (tk_ids, tk_log, tk_lse)

class Prefetcher:
    def __init__(s, files, rng):
        s.files, s.rng, s.q = files, rng, queue.Queue(maxsize=4); threading.Thread(target=s.run, daemon=True).start()
    def run(s):
        while True:
            fn = s.rng.choice(s.files)
            try: s.q.put(load_seq(fn))
            except Exception as e: print(f"[data] skip {fn}: {e}", flush=True)
    def get(s): return s.q.get()

class Pool:
    """Each loaded sequence serves ~POOL batches (different anchors each time) so NFS reads (150 MB/seq) are amortized."""
    def __init__(s, get, size):
        s.get, s.size, s.items, s.rng = get, size, [], random.Random(99)
    def __call__(s):
        s.calls = getattr(s, "calls", 0) + 1
        if len(s.items) < s.size: s.items.append(s.get())
        elif s.calls % max(1, B) == 0: s.items[s.rng.randrange(s.size)] = s.get()     # one fresh seq per batch
        return s.rng.choice(s.items)
POOL = int(E("POOL", E("REUSE", "8")))

def make_batch(src, rng):
    """B sequences -> context chunk of <=T tokens + G anchors each. Anchor a: context = rows [c0, a), query = [ids[a], mask*K]
    at positions pos[a]..pos[a]+K, labels ids[a+1..a+K]. Blocks are flattened request-major (G*BLK) so conv pos = idx & 7."""
    auxs, pcs, blks, pqs, labs, masks, ancs, sids, slog, slse = [], [], [], [], [], [], [], [], [], []
    for _ in range(B):
        item = src(); ids, pos, aux = item[:3]; soft = item[3] if len(item) > 3 else None
        n = ids.shape[0]
        if n < BLK + 2: continue
        c0 = rng.randint(0, max(0, n - T)); c1 = min(n, c0 + T); Tn = c1 - c0
        ids_c, pos_c, aux_c = ids[c0:c1], pos[c0:c1], aux[c0:c1]
        lo, hi = 1, Tn - K - 1                      # need K labels after the anchor within the chunk
        if hi < lo: continue
        anc = torch.tensor(sorted(rng.randint(lo, hi) for _ in range(G)))
        blk = torch.full((G, BLK), MASK_ID, dtype=torch.long); blk[:, 0] = ids_c[anc]
        lab = torch.stack([ids_c[a + 1:a + 1 + K] for a in anc.tolist()])            # [G,K]
        pq = pos_c[anc][:, None] + torch.arange(BLK)[None]                             # [G,BLK]
        # mask [G*BLK, T + G*BLK]: ctx key i allowed iff i < a_g and pos_i > pos_q - SW ; block keys: own block only
        qpos = pq.reshape(-1); ai = anc.repeat_interleave(BLK)
        ctx_ok = (torch.arange(T)[None] < ai[:, None]) & (torch.arange(T)[None] < Tn)
        ctx_ok &= (F.pad(pos_c, (0, T - Tn))[None] > (qpos[:, None] - SW))
        blk_ok = (torch.arange(G).repeat_interleave(BLK)[None] == torch.arange(G).repeat_interleave(BLK)[:, None])
        m = torch.cat([ctx_ok, blk_ok], 1)
        auxs.append(F.pad(aux_c, (0, 0, 0, T - Tn))); pcs.append(F.pad(pos_c, (0, T - Tn)))
        blks.append(blk.reshape(-1)); pqs.append(pq.reshape(-1)); labs.append(lab); masks.append(m); ancs.append(ids_c[anc])
        # soft target for label j (token ids[a+j]) = target logits of input row a+j-1
        lab_rows = (anc[:, None] + torch.arange(K)[None]) + c0                       # [G,K] absolute row index
        if soft is not None:
            ti, tl, tz = soft; sids.append(ti[lab_rows]); slog.append(tl[lab_rows]); slse.append(tz[lab_rows])
        else:
            sids.append(torch.full((G, K, 1), -1, dtype=torch.long)); slog.append(torch.zeros(G, K, 1)); slse.append(torch.full((G, K), float("nan")))
    if not auxs: return None
    # keys-softpad: mixed corpus (soft top-K rows from keys31 capture + hard-only rows) -> pad soft ids/logits to a common K
    Kmax = max(x.shape[-1] for x in sids)
    sids = [F.pad(x, (0, Kmax - x.shape[-1]), value=-1) for x in sids]; slog = [F.pad(x, (0, Kmax - x.shape[-1]), value=0.0) for x in slog]
    st = lambda xs: torch.stack(xs).to(DEV, non_blocking=True)
    return (st(auxs).to(torch.bfloat16), st(pcs), st(blks), st(pqs), st(labs), st(masks)[:, None], st(ancs),
            st(sids), st(slog), st(slse))

# ---------------- loss ----------------
W_POS = torch.tensor([math.exp(-(j) / GAMMA) for j in range(K)])   # j=0 -> mask slot 1
DPACE = {"reach": torch.ones(K), "n": 0}   # EMA of P(greedy chain reaches slot j) = prod_{i<j} top1_i, from training batches

def pos_weights(device):
    if POS_WEIGHT == "dpace" and DPACE["n"] > 0:
        w = DPACE["reach"].clone(); w = w / w.sum() * W_POS.sum()     # same total mass as the fixed schedule
        return w.to(device)
    return W_POS.to(device)

def dpace_update(top1):
    p = top1.cpu(); reach = torch.cumprod(torch.cat([torch.ones(1), p[:-1]]), 0)
    DPACE["reach"] = reach if DPACE["n"] == 0 else 0.98 * DPACE["reach"] + 0.02 * reach; DPACE["n"] += 1

def soft_slot_loss(lg, tk_ids, tk_log, tk_lse):
    """-sum_k p_k log q_k (- p_tail log q_tail): p from the captured target top-K (SOFT_TEMP, TAIL), q = draft full-vocab softmax.
    lg [R,V] fp32, tk_ids [R,Kt] (-1 = none), tk_log [R,Kt], tk_lse [R]. Returns per-row loss [R] and has_soft mask [R]."""
    has = tk_ids[:, 0] >= 0
    ids = tk_ids.clamp(min=0)
    logq = torch.log_softmax(lg, -1); lq_k = logq.gather(1, ids)                              # [R,Kt]
    if TAIL == "bucket" and SOFT_TEMP == 1.0:
        p_k = torch.exp(tk_log - tk_lse.nan_to_num(0.0)[:, None]); p_tail = (1 - p_k.sum(1)).clamp(min=0.0, max=1.0)
        q_tail = (1 - torch.exp(lq_k).sum(1)).clamp(min=1e-6); loss = -(p_k * lq_k).sum(1) - p_tail * torch.log(q_tail)
    else:                                                                                     # renormalize over the K candidates
        p_k = torch.softmax(tk_log / SOFT_TEMP, -1); loss = -(p_k * lq_k).sum(1)
    return torch.where(has, loss, torch.zeros_like(loss)), has

def selector_walk(S, cand):
    """Self-conditioned greedy walk (mirror of _selector_walk_kernel at temp 0): S [N,K,k,k], cand [N,K,k] -> tokens [N,K]."""
    N = S.shape[0]; ar = torch.arange(N, device=S.device); prev = torch.zeros(N, dtype=torch.long, device=S.device); out = []
    for st in range(K):
        idx = S[ar, st, prev].argmax(-1); out.append(cand[ar, st, idx]); prev = idx
    return torch.stack(out, 1)

def losses(m, batch):
    aux, pc, blk, pq, lab, mask, anc, tk_ids, tk_log, tk_lse = batch
    with torch.autocast(device_type="cuda" if DEV.startswith("cuda") else "cpu", dtype=torch.bfloat16):
        hf = m(aux, pc, blk, pq, mask)                                   # [B, G*BLK, HID]
        hf = hf.view(hf.shape[0] * G, BLK, HID)[:, 1:]                   # mask slots only [N,K,HID]
        logits = m.lm_head(hf.to(m.lm_head.weight.dtype))                # [N,K,V]
    lg = logits.float(); N = lg.shape[0]; lab = lab.reshape(N, K)
    ce = F.cross_entropy(lg.reshape(-1, VOCAB), lab.reshape(-1), reduction="none").view(N, K)
    w = pos_weights(ce.device)
    if LOSS in ("kl", "hybrid"):
        Kt = tk_ids.shape[-1]
        soft, has = soft_slot_loss(lg.reshape(-1, VOCAB), tk_ids.reshape(-1, Kt), tk_log.reshape(-1, Kt), tk_lse.reshape(-1))
        soft = soft.view(N, K); has = has.view(N, K)
        mix = soft if LOSS == "kl" else HYB_ALPHA * soft + (1 - HYB_ALPHA) * ce
        per = torch.where(has, mix, ce)                                          # rows without target logits: hard CE
        ce_w = (per * w).sum(1).mean() / w.sum()
    else:
        ce_w = (ce * w).sum(1).mean() / w.sum()
    hit_slot = (lg.argmax(-1) == lab)
    top1 = hit_slot.float().mean(0)                                      # per-slot agreement
    acc_len = hit_slot.long().cumprod(1).sum(1).float().mean()           # exact greedy accepted tokens/pass (leading correct slots)
    # selector path loss (top-k candidates from the drafter's own logits, teacher-forced true chain)
    unary, cand = torch.topk(lg.detach(), SEL_K, dim=-1)                 # [N,K,k]
    h_sel = hf.detach() if SEL_DETACH_H else hf
    with torch.autocast(device_type="cuda" if DEV.startswith("cuda") else "cpu", dtype=torch.bfloat16):
        S = m.candidate_selector.scores(cand, unary, h_sel, anc.reshape(N))   # [N,K,k,k]
    S = S.float()
    hit = (cand == lab[:, :, None]); idx = hit.float().argmax(-1); has = hit.any(-1)       # true index per step
    prev = torch.cat([torch.zeros(N, 1, dtype=torch.long, device=idx.device), idx[:, :-1]], 1)
    valid = has & torch.cat([torch.ones(N, 1, dtype=torch.bool, device=has.device), has[:, :-1]], 1)
    Srow = S.gather(2, prev[:, :, None, None].expand(-1, -1, 1, SEL_K)).squeeze(2)          # [N,K,k] scores of true predecessor row
    if SEL_TARGET == "soft" and tk_ids.shape[-1] > 1:
        # target mass on the draft's candidates: p(cand_c) looked up in the captured top-K (0 if absent), renormalized; hard fallback
        Kt = tk_ids.shape[-1]; ti = tk_ids.reshape(N, K, Kt); tl = tk_log.reshape(N, K, Kt)
        p_full = torch.softmax(tl / SOFT_TEMP, -1) * (ti[:, :, 0:1] >= 0)
        match = (cand[:, :, :, None] == ti[:, :, None, :])                                  # [N,K,k,Kt]
        p_c = (match * p_full[:, :, None, :]).sum(-1); mass = p_c.sum(-1, keepdim=True)
        soft_ok = (mass.squeeze(-1) > 0.05) & (ti[:, :, 0] >= 0)
        tgt = torch.where(soft_ok[..., None], p_c / mass.clamp(min=1e-9), F.one_hot(idx, SEL_K).float())
        sel_ce = -(tgt * torch.log_softmax(Srow, -1)).sum(-1)
    else:
        sel_ce = F.cross_entropy(Srow.reshape(-1, SEL_K), idx.reshape(-1), reduction="none").view(N, K)
    if SEL_POSW:  # share the block-loss position weights with the selector loss (reference recipes); default off = ft4 behaviour
        wv = (valid.float() * w[None]); sel_loss = (sel_ce * wv).sum() / wv.sum().clamp(min=1e-6)
    else:
        sel_loss = (sel_ce * valid).sum() / valid.sum().clamp(min=1)
    chain_acc = ((Srow.argmax(-1) == idx) & valid).float().sum(0) / valid.float().sum(0).clamp(min=1)
    walk = selector_walk(S, cand); acc_len_walk = (walk == lab).long().cumprod(1).sum(1).float().mean()   # self-conditioned greedy walk
    return ce_w, sel_loss, top1.detach(), chain_acc.detach(), has.float().mean().item(), acc_len.item(), acc_len_walk.item()

# ---------------- main ----------------
def main():
    t0 = time.time(); m = build()
    m = apply_train_set(m).to(DEV)
    if MODE == "roundtrip":   # real weights in -> incoai layout out, no data (CPU-safe smoke of the load/export contract)
        save(m, OUT); return
    if MODE == "synth":
        rng = random.Random(SEED); src = Pool(lambda: synth_seq(rng), POOL); hold = lambda: synth_seq(rng)
    else:
        tr, ho = list_seqs(); print(f"[data] train seqs={len(tr)} holdout={len(ho)} from {SEQ_DIR}", flush=True)
        if not tr: print("NO DATA — run capture + assemble first"); return
        pf = Prefetcher(tr, random.Random(SEED)); src = Pool(pf.get, POOL)
        hold_cache = [load_seq(f) for f in ho[:int(E("HOLDOUT_MAX", "64"))]]
    rng = random.Random(SEED)

    def evaluate(tag, nb=None):
        """FIXED holdout: every holdout sequence, EVAL_CHUNKS chunks each, anchors from Random(1000+i) -> identical batches on
        every call (the old version drew 8 sequences from a stateful RNG, so consecutive evals scored different sequences)."""
        m.eval(); acc = []; saveB = globals()["B"]; globals()["B"] = 1
        with torch.no_grad():
            for i, sq in enumerate(hold_cache if MODE != "synth" else [synth_seq(random.Random(500 + j)) for j in range(8)]):
                r = random.Random(1000 + i)
                for _ in range(int(E("EVAL_CHUNKS", "2"))):
                    b = make_batch(lambda: sq, r)
                    if b is None: continue
                    ce_w, sl, top1, chain, cov, al, aw = losses(m, b); nblk = b[4].shape[0] * b[4].shape[1]
                    acc.append((ce_w.item(), sl.item(), top1.cpu(), chain.cpu(), cov, nblk, al, aw))
        globals()["B"] = saveB; m.train()
        if acc:
            wts = torch.tensor([float(a[5]) for a in acc]); wts = wts / wts.sum()
            ce = sum(a[0] * w for a, w in zip(acc, wts.tolist())); sl = sum(a[1] * w for a, w in zip(acc, wts.tolist()))
            top1 = (torch.stack([a[2] for a in acc]) * wts[:, None]).sum(0); chain = (torch.stack([a[3] for a in acc]) * wts[:, None]).sum(0)
            # expected accepted tokens per pass under independent-slot approximation: sum_j prod_{i<=j} p_i
            p = top1.tolist(); exp_acc = sum(math.prod(p[:j + 1]) for j in range(K))
            print(f"[{tag}] ce_w {ce:.3f} sel {sl:.3f} cov {sum(a[4]*w for a, w in zip(acc, wts.tolist())):.2f} "
                  f"top1/slot {[round(x, 3) for x in p]} chain/slot {[round(x, 3) for x in chain.tolist()]} "
                  f"~E[accept] {exp_acc:.2f} greedy acc_len {sum(a[6]*w for a, w in zip(acc, wts.tolist())):.3f} "
                  f"selector-walk acc_len {sum(a[7]*w for a, w in zip(acc, wts.tolist())):.3f} (blocks {sum(a[5] for a in acc)})", flush=True)
            return exp_acc
    evaluate("parity-before")
    if MODE == "parity": return
    EVAL_EVERY = int(E("EVAL_EVERY", "100"))
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=WD, foreach=True)
    warm = int(E("WARMUP_STEPS", "0")) or max(1, int(STEPS * WARMUP))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, s - warm) / max(1, STEPS - warm)))))
    m.train(); hist = []
    for step in range(STEPS):
        b = make_batch(src, rng)
        if b is None: continue
        ce_w, sel, top1, chain, cov, al, aw = losses(m, b)
        if POS_WEIGHT == "dpace": dpace_update(top1)
        loss = ce_w + SEL_W * sel
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        hist.append((ce_w.item(), sel.item()))
        if step % LOG_EVERY == 0 or step == STEPS - 1:
            r = hist[-LOG_EVERY:]; mem = torch.cuda.max_memory_allocated() / 2**30 if DEV.startswith("cuda") else 0
            print(f"[train] step {step:5d} {LOSS} {ce_w.item():.3f} sel {sel.item():.3f} acc_len {al:.2f}/walk {aw:.2f} (ma {sum(x[0] for x in r)/len(r):.3f}/{sum(x[1] for x in r)/len(r):.3f}) "
                  f"top1 {[round(x,2) for x in top1.tolist()]} chain {[round(x,2) for x in chain.tolist()]} gn {gn:.2f} lr {sched.get_last_lr()[0]:.2e} "
                  f"peak {mem:.1f}G {time.time()-t0:.0f}s", flush=True)
        if EVAL_EVERY and (step + 1) % EVAL_EVERY == 0 and step + 1 < STEPS:
            evaluate(f"holdout@{step+1}")
        if SAVE_EVERY and (step + 1) % SAVE_EVERY == 0 and step + 1 < STEPS:
            save(m, f"{OUT}/step{step+1}")
    evaluate("holdout-final"); sd = save(m, OUT)
    if MODE == "synth":  # round-trip check on the synthetic layout
        from safetensors.torch import load_file
        back = load_file(f"{OUT}/model.safetensors"); assert set(back) == set(sd) and all(back[k].shape == sd[k].shape for k in sd)
        print("[synth] save/load round-trip OK; first/last ce", hist[0][0], "->", hist[-1][0], flush=True)

if __name__ == "__main__":
    main()
