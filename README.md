# Full GLM-5.3 on four DGX Sparks with TensorFold (tensor parallel 4)

The full **GLM-5.3** (753B total, ~40B active, `glm_moe_dsa`) served natively by [TensorFold](https://github.com/ashhart/TensorFold)
across **four NVIDIA DGX Sparks** (GB10, 128 GB unified memory each), one rank per Spark over their ConnectX-7 RoCE
fabric — TensorFold's first tensor-parallel-4 engine. No vLLM in the serving path, and TensorFold's guarantee holds: a
drafted reply is bit-identical to a serial one.

- Code: [ashhart/TensorFold#159](https://github.com/ashhart/TensorFold/pull/159) (branch
  [`drowzeys/TensorFold:glm-moe-dsa-tp4`](https://github.com/drowzeys/TensorFold/tree/glm-moe-dsa-tp4))
- Weights: [drowzeys/keys-GLM-5.3-EXL3-2.75BPW](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW)
  (EXL3, a bit width per routed expert averaging 2.75 bpw; KL 0.124 nats / top-1 89.6 % vs BF16 over 65,536 tokens)
- How to run it: [RECIPE.md](RECIPE.md)

## Results

Four DGX Sparks, single stream, 32K-token context, temperature 1.0 / top-p 0.95, 512 tokens. The vLLM row is the same
checkpoint on the same four Sparks with a tuned vLLM (EXL3 kernels, MTP k = 2, CUDA graphs, RoCE all-reduce).

| Engine | Prose | Code | Step | Prefill 8K / 32K | TTFT 32K |
|---|---|---|---|---|---|
| TensorFold, MTP k = 2 (default) | **24.6 tok/s** | 27.3 tok/s | 78.5 ms | **773-783** / 705 tok/s | 46 s |
| TensorFold, DFlash2 drafts (depth 7, confidence 0.4) | 23.0 | **32.4** | 87-101 ms | | |
| TensorFold, auto (MTP or DFlash2 each round) | 23.6 | 27.6 | 88-90 ms | | |
| vLLM, same checkpoint | 23.9 | 30.3 | 77-78 ms | ~755 / 749 tok/s | 43.6 s |

Short context, greedy:

| | Prose | Code |
|---|---|---|
| MTP | 28.8 tok/s | 31.7 tok/s |
| DFlash2 | 28.3 | **40.0** (3.6 tokens a verify step) |
| auto | **30.9** | 32.6 |

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
