# Full GLM-5.3 on four DGX Sparks with TensorFold (tensor parallel 4)

The full **GLM-5.3** (753B total, ~40B active) served natively by [TensorFold](https://github.com/ashhart/TensorFold)
across **four NVIDIA DGX Sparks** (GB10, one rank per Spark over ConnectX-7 RoCE). No vLLM in the serving path.
Image `ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-04` (TensorFold 0.6.5 + this engine).

## Performance (image `2026-10-04`)

Four DGX Sparks, full GLM-5.3 at 2.75 bpw (EXL3), default settings, the public
[incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) draft. Decode: means of 3 repeats.

| | Speed | Detail |
|---|---|---|
| **32K context, prose** (T = 1.0) | **29.2 tok/s** | DFlash2, 2.20 tokens a round, 75 ms a round |
| **32K context, code** (T = 1.0) | **34.5 tok/s** | DFlash2, 2.84 tokens a round, 83 ms |
| **Short context, code** (greedy) | **40.2 tok/s** | DFlash2, 3.32 tokens a round |
| **Short context, prose** (greedy) | **31.4 tok/s** | DFlash2, 2.25 tokens a round |
| MTP only (no draft model) | 27.3 prose / 29.9 code at 32K · 31.1 / 35.1 short | 62.6 ms a round short, 69.7 ms at 32K ¹ |
| **Prefill** | **8K 1,045 · 32K 1,115 · 128K 1,000 tok/s** | |
| **TTFT** | **8K 7.8 s · 32K 29.3 s · 128K 130 s** | needles pass at 32K and 128K |
| **4 concurrent streams** | **69.7 chat / 95.4 code tok/s** aggregate (greedy) | TTFT ≤ 1.4 s; replies equal the same request alone ¹ |
| 1M context | needles pass at 128K / 512K / ~1M | decode context parallelism 4 (validated on `2026-09-30`) |

¹ measured on `2026-10-03` (same serving code; `2026-10-04` adds the TensorFold 0.6.5 merge).

### Against the previous image and vLLM (32K context, sampled)

| | Prose | Code | Prefill 8K / 32K | TTFT 32K / 128K |
|---|---|---|---|---|
| **TensorFold `2026-10-04`** | **29.2 tok/s** | **34.5 tok/s** | **1,045 / 1,115 tok/s** | **29 s / 130 s** |
| TensorFold `2026-09-30` | 24.6 | 32.4 | 773-783 / 705 | 46 s / 262 s |
| vLLM, same checkpoint (EXL3 kernels, MTP k = 2, RoCE) | 23.9 | 30.3 | ~755 / 749 | 43.6 s / — |

**+19 % prose, +58 % prefill at 32K, half the 128K TTFT** compared with `2026-09-30`.

## What changed

**`2026-10-04`** — TensorFold **0.6.5** merged in (upstream now serves token-id prompts itself; no speed change),
test fixes, and a cache-fit guard that no longer over-reserves on small or shared GPUs.

**`2026-10-03`** (since `2026-09-30`):

| Change | Effect |
|---|---|
| Grouped EXL3 decode linears (q_a + kv_a, wq_b + q_b, gate + up in one launch), PDL launches; tiles tuned at load and **every rank takes rank 0's** (per-rank picks let the slowest pace every layer) | 78.5 → 69.7 ms a round at 32K |
| Indexer top-k: radix select for prompt chunks, `torch.topk` for decode windows | faster prefill; decode keeps its speed |
| Sequence-parallel prompt chunks, routed prompt experts for any K, per-shape prompt GEMM tiles, 8192-row chunks for long prompts | prefill 705 → ~1,100 tok/s at 32K; 128K TTFT 262 → 130 s |
| Concurrent streams (`--parallel N`) with MTP or DFlash2 drafts per stream; short prompts fill whole | 4 streams ~70-95 tok/s aggregate, TTFT ≤ 1.4 s |
| Reproducible long prompts on request: `TF_EXL3_PROMPT_DET=slots16` (see below) | same reply every run, ~4-6 % slower prefill |
| The runner refuses a `--context` × `--parallel` whose caches would not fit | GB10's unified memory swaps instead of failing |
| `one-shot.sh` looks up each Spark's RoCE GID (they move across reboots); `PARALLEL`, `DOCKER_ENV` | |

## Quick start (prebuilt image)

```sh
# on every Spark, once per boot
sudo sysctl -w vm.compaction_proactiveness=0
# from any machine that can ssh to the four Sparks
NODES="spark1 spark2 spark3 spark4" MODEL=/models/GLM-5.3-EXL3-2.75BPW DRAFT=/models/GLM-5.3-DFlash2 ./one-shot.sh up
curl http://spark1:8890/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "glm-5.3-tf", "messages": [{"role": "user", "content": "Hello"}], "tf_mtp": "dflash"}'
```

Four concurrent streams: `PARALLEL=4 CONTEXT=32768 DOCKER_ENV="-e TF_GLM53_CONC_MODE=dflash" ./one-shot.sh up`.
Requests pick the draft with `"tf_mtp"`: `"normed/normed"` (MTP, default), `"dflash"` or `"auto"`.

- Code: [ashhart/TensorFold#159](https://github.com/ashhart/TensorFold/pull/159) (branch
  [`drowzeys/TensorFold:glm-moe-dsa-tp4`](https://github.com/drowzeys/TensorFold/tree/glm-moe-dsa-tp4))
- Weights: [drowzeys/keys-GLM-5.3-EXL3-2.75BPW](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW)
  (EXL3, a bit width per routed expert averaging 2.75 bpw; KL 0.124 nats / top-1 89.6 % vs BF16 over 65,536 tokens)
- From source: [RECIPE.md](RECIPE.md)

## Reproducibility

Decode is row-invariant (rank-order sums, no atomics): a drafted reply equals a serial one, and concurrent replies
equal the same request alone. Prompts longer than one chunk (~4K tokens) go through the MoE prompt kernel, which by
default adds a row's expert outputs with fp32 atomics in arrival order — the fastest prefill, but the same long prompt
can give a different reply run to run. For reproducible long prompts:

```sh
DOCKER_ENV="-e TF_EXL3_PROMPT_DET=slots16" ./one-shot.sh up   # fp16 expert rows summed in a fixed order
```

| `TF_EXL3_PROMPT_DET` | Same reply every run | Prefill 8K / 32K / 128K |
|---|---|---|
| `0` (default) | short prompts only | 1,045 / 1,115 / 1,000 tok/s |
| `slots16` | yes (checked at 8K and 32K, serial and DFlash2) | 1,000 / 969 / 952 tok/s |

A DFlash2 draft fine-tuned on-policy against this checkpoint reaches 30.6 prose / 37.0 code at 32K, but it derives
from a CC-BY-NC-ND draft and is not redistributed.

## Node notes that cost us time

- `vm.compaction_proactiveness=0` on every node (compaction stalls a rank ~130 ms at a time).
- RoCE GID indices can move across reboots: resolve each node's RoCE v2 GID for its fabric IPv4 at launch
  (`one-shot.sh` does). After a full-cluster reboot, ping the fabric peers before the first boot.
- Anything else on a rank — a desktop browser, other containers — shows up as rank skew: all four wait for the
  slowest at every all-reduce. Keep the Sparks clean.
- `--parallel 4` needs a smaller context (each stream holds its own cache): `--context 32768` fits.
- A node that reboots mid-CUDA-extension build leaves a lock under the extension cache; delete it before the next start.

## This repo

- [RECIPE.md](RECIPE.md) — how to run it from source, the node settings that matter.
- [one-shot.sh](one-shot.sh) — launcher for the prebuilt image from any machine.
- [node/gb10-node-settings.sh](node/gb10-node-settings.sh) — the runtime setting to apply on every node.
- [bench/](bench/) — the benches behind the tables; [evidence/](evidence/) — run logs and raw results.

## Credits

[Z.ai](https://huggingface.co/zai-org) (GLM-5.3) · [ashhart / TensorFold](https://github.com/ashhart/TensorFold) (the
engine this builds on, MIT) · [turboderp / exllamav3](https://github.com/turboderp-org/exllamav3) (EXL3) · vcruz305
(per-expert mixed-width EXL3 work the quantization builds on) · cuda-exl3 (the grouped prompt GEMM) ·
[b12x](https://github.com/local-inference-lab/b12x) (RoCEnante one-shot all-reduce) · MiaAI-Lab (GLM prompt-kernel
designs the prompt experts adapt) · incoai (the DFlash2 draft).

Scripts here: MIT (see LICENSE). Model weights: the GLM-5.3 license of the base model.
