#!/usr/bin/env python3
"""Fetch Blackfrost-derisked BF16/F16 tensors from keys-GLM-5.3-EXL3-Abliterated
and encode them as EXL3 K=5 mul1 for the TensorFold 2.75 mixed-K overlay.

The 2.75BPW champion stores non-expert linears as EXL3 K5, so the 3bpw ablit pack's
native F16 o_proj / down_proj cannot be dropped in. This re-quantizes the same
derisked tensors (layers 2-49, MTP and L0-1 left stock).

Usage (inside glm53-exl3-quant:fork150):
  python3 encode_ablit_k5.py --out /work/k5 --layers 2-25
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import save_file
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

REPO = "drowzeys/keys-GLM-5.3-EXL3-Abliterated"
INDEX_URL = f"https://huggingface.co/{REPO}/resolve/main/model.safetensors.index.json"
RESOLVE = f"https://huggingface.co/{REPO}/resolve/main/"


def token() -> str:
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if t:
        return t.strip()
    p = Path("/root/.cache/huggingface/token")
    if p.exists():
        return p.read_text().strip()
    p = Path.home() / ".cache/huggingface/token"
    return p.read_text().strip()


def opener(tok: str) -> urllib.request.OpenerDirector:
    o = urllib.request.build_opener()
    o.addheaders = [
        ("Authorization", f"Bearer {tok}"),
        ("User-Agent", "keys-glm53-tf-ablit/1.0"),
    ]
    return o


def http_range(op: urllib.request.OpenerDirector, url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with op.open(req, timeout=300) as r:
        data = r.read()
    if len(data) != end - start + 1:
        raise RuntimeError(f"range {start}-{end} got {len(data)} bytes from {url}")
    return data


def load_index(op: urllib.request.OpenerDirector, cache: Path) -> dict:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return json.loads(cache.read_text())
    req = urllib.request.Request(INDEX_URL)
    with op.open(req, timeout=120) as r:
        raw = r.read()
    cache.write_bytes(raw)
    return json.loads(raw)


_header_cache: dict[str, tuple[dict, int]] = {}


def shard_header(op: urllib.request.OpenerDirector, shard: str) -> tuple[dict, int]:
    if shard in _header_cache:
        return _header_cache[shard]
    url = RESOLVE + shard
    n = struct.unpack("<Q", http_range(op, url, 0, 7))[0]
    meta = json.loads(http_range(op, url, 8, 8 + n - 1))
    data_start = 8 + n
    _header_cache[shard] = (meta, data_start)
    return meta, data_start


def fetch_f16(op: urllib.request.OpenerDirector, shard: str, key: str) -> torch.Tensor:
    meta, data_start = shard_header(op, shard)
    info = meta[key]
    if info["dtype"] != "F16":
        raise RuntimeError(f"{key} dtype {info['dtype']} (want F16)")
    a, b = info["data_offsets"]
    raw = http_range(op, RESOLVE + shard, data_start + a, data_start + b - 1)
    t = torch.frombuffer(bytearray(raw), dtype=torch.float16).clone()
    t = t.view(*info["shape"])
    return t


def identity_H(in_f: int, device: torch.device, key: str) -> dict:
    # meta H triggers uncalibrated (data-free) fallback, same as convert.py with no capture
    return {
        "H": torch.empty(in_f, in_f, device="meta"),
        "first_key": key,
        "count": 0,
        "finalized": False,
        "num_total": 0,
        "inf_nan": torch.zeros(2, dtype=torch.long),
        "device": device,
    }


def encode_linear(W_out_in: torch.Tensor, K: int, key: str, device: torch.device, seed: int) -> dict:
    """W_out_in is F16/BF16 [out, in] (HF / torch layout). Returns EXL3 packed tensors on CPU."""
    W = W_out_in.t().contiguous().float()  # (in, out) as quantize_exl3 expects
    in_f, out_f = W.shape
    qa = {
        "seed": seed,
        "K": K,
        "devices": [str(device)],
        "apply_out_scales": True,
        "mul1": True,
    }
    t0 = time.time()
    _, proxy, out = quantize_exl3(W, identity_H(in_f, device, key), qa, False, None, False)
    dt = time.time() - t0
    packed = {k: v.detach().to("cpu") if torch.is_tensor(v) else v for k, v in out.items()}
    packed["_meta"] = {
        "key": key,
        "in_features": in_f,
        "out_features": out_f,
        "K": K,
        "proxy_err": float(proxy) if proxy is not None else None,
        "q_fallback": bool(qa.get("q_fallback")),
        "seconds": round(dt, 3),
        "trellis": list(packed["trellis"].shape),
        "suh": list(packed["suh"].shape),
        "svh": list(packed["svh"].shape),
    }
    return packed


def layer_jobs(lo: int, hi: int) -> list[tuple[int, str, str]]:
    """(layer, kind, hf_key) kind in {o_proj, shared_down, dense_down}."""
    jobs = []
    for L in range(lo, hi + 1):
        jobs.append((L, "o_proj", f"model.layers.{L}.self_attn.o_proj"))
        if L == 2:
            jobs.append((L, "dense_down", f"model.layers.{L}.mlp.down_proj"))
        elif L >= 3:
            jobs.append((L, "shared_down", f"model.layers.{L}.mlp.shared_experts.down_proj"))
    return jobs


def out_name(kind: str, L: int) -> str:
    return f"L{L:02d}.{kind}.safetensors"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="2-49", help="inclusive, e.g. 2-25")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    a, _, b = args.layers.partition("-")
    lo, hi = int(a), int(b or a)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = token()
    op = opener(tok)
    idx = load_index(op, out / "hf_index.json")
    wm = idx["weight_map"]
    device = torch.device(args.device)
    jobs = layer_jobs(lo, hi)
    report = []
    for i, (L, kind, prefix) in enumerate(jobs, 1):
        dest = out / out_name(kind, L)
        if dest.exists():
            print(f"[{i}/{len(jobs)}] skip {dest.name}", flush=True)
            continue
        hf_key = prefix + ".weight"
        shard = wm[hf_key]
        print(f"[{i}/{len(jobs)}] fetch {hf_key} from {shard}", flush=True)
        t0 = time.time()
        W = fetch_f16(op, shard, hf_key)
        print(f"  fetched {tuple(W.shape)} {W.dtype} in {time.time()-t0:.1f}s, encode K={args.K}", flush=True)
        packed = encode_linear(W, args.K, prefix, device, seed=L)
        meta = packed.pop("_meta")
        save_file(packed, str(dest))
        (out / (dest.name + ".json")).write_text(json.dumps(meta, indent=2))
        print(f"  wrote {dest.name} trellis={meta['trellis']} proxy={meta['proxy_err']}  {meta['seconds']}s", flush=True)
        report.append(meta)
        del W, packed
        torch.cuda.empty_cache()
    (out / f"report_{lo}_{hi}.json").write_text(json.dumps(report, indent=2))
    print("DONE", len(report), "encoded", lo, hi, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
