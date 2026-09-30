# Full GLM-5.3 on four DGX Sparks with TensorFold (tensor parallel 4)

The full **GLM-5.3** (753B total, ~40B active, `glm_moe_dsa`) served natively by [TensorFold](https://github.com/ashhart/TensorFold)
across **four NVIDIA DGX Sparks** (GB10, 128 GB unified memory each), one rank per Spark over their ConnectX-7 RoCE
fabric — TensorFold's first tensor-parallel-4 engine. No vLLM in the serving path, and TensorFold's guarantee holds: a
drafted reply is bit-identical to a serial one.

- Code: [ashhart/TensorFold#159](https://github.com/ashhart/TensorFold/pull/159) (branch
  [`drowzeys/TensorFold:glm-moe-dsa-tp4`](https://github.com/drowzeys/TensorFold/tree/glm-moe-dsa-tp4))
- Weights: [drowzeys/keys-GLM-5.3-EXL3-2.75BPW](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW)
  (EXL3, a bit width per routed expert averaging 2.75 bpw; KL 0.124 nats / top-1 89.6 % vs BF16 over 65,536 tokens)
- How to run it: [one-shot.sh](one-shot.sh) with the prebuilt image (below), or [RECIPE.md](RECIPE.md) from source

## Quick start (prebuilt image)

```sh
# on every Spark, once per boot
sudo sysctl -w vm.compaction_proactiveness=0
# from any machine that can ssh to the four Sparks
NODES="spark1 spark2 spark3 spark4" MODEL=/models/GLM-5.3-EXL3-2.75BPW ./one-shot.sh up
#   + DRAFT=/models/GLM-5.3-DFlash2 for DFlash2 / auto drafts
curl http://spark1:8890/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "glm-5.3-tf", "messages": [{"role": "user", "content": "Hello"}], "tf_mtp": "dflash"}'
```

Image: `ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-09-30` (CUDA 13 / GB10; TensorFold `glm_moe_dsa`
with its CUDA extensions prebuilt, cuda-exl3, b12x RoCE; the fastest settings as defaults).

## Best configurations at a glance

Four DGX Sparks, single stream, full GLM-5.3 (2.75 bpw EXL3). Pick the draft by workload:

| Workload | Best configuration | Speed | Tokens / verify step |
|---|---|---|---|
| **Short context, prose (greedy)** | `"tf_mtp": "auto"` (MTP or DFlash2 each round) | **30.9 tok/s** | 2.36 |
| **Short context, code (greedy)** | `"tf_mtp": "dflash"` (depth 7, confidence 0.4) | **40.0 tok/s** | 3.6 |
| 32K context, prose (T = 1.0) | default MTP (`normed/normed`) | 24.6 tok/s | 1.94 |
| 32K context, code (T = 1.0) | `"tf_mtp": "dflash"` | 32.4 tok/s | 3.26 |
| Prefill / TTFT | 4K: 629-698 tok/s, 5.9-6.5 s · **8K: 773-783 tok/s, 10.4 s** · 32K: 705 tok/s, 46 s | | |
| Context | one sequence at a time; `--context 36864` tested (~250-300K tokens estimated to fit; 1M needs decode context parallelism, not built yet) | | |

## Short context, greedy

| Draft | Prose | Code |
|---|---|---|
| MTP k = 2 (default) | 28.8 tok/s | 31.7 tok/s |
| DFlash2 (depth 7, confidence 0.4) | 28.3 | **40.0** |
| auto | **30.9** | 32.6 |

## 32K context, sampled (temperature 1.0 / top-p 0.95, 512 tokens)

The vLLM row is the same checkpoint on the same four Sparks with a tuned vLLM (EXL3 kernels, MTP k = 2, CUDA graphs,
RoCE all-reduce).

| Engine | Prose | Code | Step | Prefill 8K / 32K | TTFT 32K |
|---|---|---|---|---|---|
| TensorFold, MTP k = 2 (default) | **24.6 tok/s** | 27.3 tok/s | 78.5 ms | **773-783** / 705 tok/s | 46 s |
| TensorFold, DFlash2 | 23.0 | **32.4** | 87-101 ms | | |
| TensorFold, auto | 23.6 | 27.6 | 88-90 ms | | |
| vLLM, same checkpoint | 23.9 | 30.3 | 77-78 ms | ~755 / 749 tok/s | 43.6 s |

For 1M-token contexts today, the vLLM path with decode context parallelism 4 serves a 1.23M-token KV pool at 16.6 prose
/ 21.1 code tok/s; the TensorFold engine does not do 1M yet.

DFlash2 drafts: [incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) (CC-BY-NC-ND-4.0), trained
against BF16 GLM-5.3 — on the 2.75 bpw target its prose acceptance is modest; code gains most.

## What made the difference on GB10

| Change | Effect |
|---|---|
| Fused, row-invariant kernels + CUDA-graph decode rounds | 310 → 97 ms a round |
| One-shot RoCE all-reduce (rank-order sums, same bits as NCCL) for decode windows | NCCL all-gathers 96 → 68 µs |
| Retiled EXL3 linears, reduced-vocabulary MTP draft head, tuned attention tilings | ~97 → 72 ms a round |
| `vm.compaction_proactiveness=0` on every node | a quarter of rounds stalled ~130 ms by memory compaction: 115 → 72 ms mean |
| MTP reads the final-normed target hidden (like vLLM) | 1.6 → 2.15 tokens a step |
| Prompt path: 4096-row equal chunks, EXL3 decode-once GEMM, cuda-exl3 grouped experts sharing one weight copy, bf16 NCCL all-reduce | 175 → ~780 tok/s at 8K |
| `NCCL_MAX_NCHANNELS=2` + two micro-batches per chunk | lets the prompt all-reduce run under compute (~2/3 hidden) |
| DFlash2 with a confidence cutoff (stop the draft chain when the drafter is unsure) | code 31.7 → 40.0 tok/s (greedy) |

## This repo

- [RECIPE.md](RECIPE.md) — how to run it (four Sparks, one launcher from any machine), the node settings that matter.
- [node/gb10-node-settings.sh](node/gb10-node-settings.sh) — the runtime setting to apply on every node.
- [bench/](bench/) — the benches behind the tables (TensorFold single-stream decode, vLLM decode, prefill/TTFT).

## Credits

[Z.ai](https://huggingface.co/zai-org) (GLM-5.3) · [ashhart / TensorFold](https://github.com/ashhart/TensorFold) (the
engine this builds on, MIT) · [turboderp / exllamav3](https://github.com/turboderp-org/exllamav3) (EXL3) · vcruz305
(per-expert mixed-width EXL3 work the quantization builds on) · cuda-exl3 (the grouped prompt GEMM) ·
[b12x](https://github.com/local-inference-lab/b12x) (RoCEnante one-shot all-reduce) · incoai (the DFlash2 draft).

Scripts here: MIT (see LICENSE). Model weights: the GLM-5.3 license of the base model.
