# Full GLM-5.3 on four DGX Sparks with TensorFold (tensor parallel 4)

The full **GLM-5.3** (753B total, ~40B active, `glm_moe_dsa`) served natively by [TensorFold](https://github.com/ashhart/TensorFold)
across **four NVIDIA DGX Sparks** (GB10, 128 GB unified memory each), one rank per Spark over their ConnectX-7 RoCE
fabric — TensorFold's first tensor-parallel-4 engine, no vLLM in the serving path. A drafted reply equals a serial
one and concurrent replies equal the same request alone; see [Reproducibility](#reproducibility) for long prompts.

- Code: [ashhart/TensorFold#159](https://github.com/ashhart/TensorFold/pull/159) (branch
  [`drowzeys/TensorFold:glm-moe-dsa-tp4`](https://github.com/drowzeys/TensorFold/tree/glm-moe-dsa-tp4), rebased on 0.6.1)
- Weights: [drowzeys/keys-GLM-5.3-EXL3-2.75BPW](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW)
  (EXL3, a bit width per routed expert averaging 2.75 bpw; KL 0.124 nats / top-1 89.6 % vs BF16 over 65,536 tokens)
- How to run it: [one-shot.sh](one-shot.sh) with the prebuilt image (below), or [RECIPE.md](RECIPE.md) from source

## Quick start (prebuilt image)

```sh
# on every Spark, once per boot
sudo sysctl -w vm.compaction_proactiveness=0
# from any machine that can ssh to the four Sparks
NODES="spark1 spark2 spark3 spark4" MODEL=/models/GLM-5.3-EXL3-2.75BPW DRAFT=/models/GLM-5.3-DFlash2 ./one-shot.sh up
curl http://spark1:8890/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "glm-5.3-tf", "messages": [{"role": "user", "content": "Hello"}], "tf_mtp": "dflash"}'
```

Image: `ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-03` (CUDA 13 / GB10; TensorFold `glm_moe_dsa`
with its CUDA extensions prebuilt, cuda-exl3, b12x RoCE; the fastest settings are the defaults — no flags needed).
`one-shot.sh` looks up each Spark's RoCE v2 GID itself (they can move across reboots).

## Best configurations at a glance

Four DGX Sparks, full GLM-5.3 (2.75 bpw EXL3), image `2026-10-03`, default settings, the public
[incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) draft. Means of 3 repeats.

| Workload | Configuration | Speed | Tokens / verify step |
|---|---|---|---|
| **Short context, code (greedy)** | `"tf_mtp": "dflash"` | **40.0 tok/s** | 3.41 |
| **Short context, prose (greedy)** | `"tf_mtp": "dflash"` | **31.0 tok/s** | 2.23 |
| Short context, MTP (greedy) | default | 31.1 prose / 35.1 code | 62.6 ms a round |
| **32K context, prose (T = 1.0)** | `"tf_mtp": "dflash"` | **27.7 tok/s** | 2.12 |
| **32K context, code (T = 1.0)** | `"tf_mtp": "dflash"` | **34.3 tok/s** | 2.99 |
| 32K context, MTP (T = 1.0) | default | 27.3 prose / 29.9 code | 69.7 ms a round |
| **Prefill / TTFT** | 4K 922 tok/s (4.5 s) · 8K 1,044-1,087 (7.5-7.8 s) · 16K 1,089 (15.0 s) · **32K 1,035-1,124 (29-31 s)** · **128K 996 (131 s)** | | |
| **4 concurrent streams** | `--parallel 4 --context 32768`, DFlash2 drafts | **69.7 chat / 95.4 code tok/s** aggregate (greedy), TTFT ≤ 1.4 s | |
| 1M context | `--context 1000000` (decode context parallelism 4) | needles pass at 128K / 512K / ~1M (previous image; not re-run on `2026-10-03`) | |

Needles (passphrase at half depth) pass at 32K and 128K on this build. Concurrent replies are checked equal to the
same request alone (`tools/bench_concurrent.py --alone`: 63/63).

## Compared with the previous image and vLLM (32K context, sampled)

| | Prose | Code | Round | Prefill 8K / 32K | TTFT 32K / 128K |
|---|---|---|---|---|---|
| **TensorFold `2026-10-03`**, DFlash2 | **27.7 tok/s** | **34.3** | 78 / 89 ms | **1,044-1,087 / 1,035-1,124** | **29-31 s / 131 s** |
| TensorFold `2026-10-03`, MTP | 27.3 | 29.9 | **69.7 ms** | | |
| TensorFold `2026-09-30` (v0.5.0), MTP / DFlash2 | 24.6 / 23.0 | 27.3 / 32.4 | 78.5 / 87-101 ms | 773-783 / 705 | 46 s / 262 s |
| vLLM, same checkpoint (EXL3 kernels, MTP k = 2, RoCE) | 23.9 | 30.3 | 77-78 ms | ~755 / 749 | 43.6 s / — |

Short context, greedy code: 40.0 tok/s on both images. A DFlash2 draft fine-tuned on-policy against this
checkpoint reaches 30.6 prose / 37.0 code at 32K (+10 % prose), but it derives from a CC-BY-NC-ND draft and is not
redistributed.

## Reproducibility

Decode is row-invariant (rank-order sums, no atomics), so a drafted reply equals a serial one and concurrent replies
equal the same request alone. Prompts longer than one prompt chunk (~4K tokens) go through the MoE prompt kernel,
which by default adds a row's expert outputs with fp32 atomics in arrival order: the fastest prefill, but the same
long prompt can give a different reply run to run (greedy replies diverged after 12-108 tokens). For reproducible
long prompts:

```sh
DOCKER_ENV="-e TF_EXL3_PROMPT_DET=slots16" ./one-shot.sh up   # fp16 expert rows summed in a fixed order
```

| `TF_EXL3_PROMPT_DET` | Same reply every run | Prefill 8K / 32K / 128K |
|---|---|---|
| `0` (default) | short prompts only | 1,044-1,087 / 1,035-1,124 / 996 tok/s |
| `slots16` | yes (checked at 8K and 32K, serial and DFlash2) | 1,000 / 969 / 952 tok/s |

## What changed since `2026-09-30`

| Change | Effect |
|---|---|
| Rebased on TensorFold 0.6.1 | — (the rebase itself costs nothing) |
| Grouped EXL3 decode linears (q_a + kv_a, wq_b + q_b, gate + up in one launch), PDL launches; tiles tuned at load and **every rank takes rank 0's** (per-rank picks let the slowest pace every layer) | 78.5 → 69.7 ms a round at 32K |
| Indexer top-k: radix select for prompt chunks, `torch.topk` for decode windows | prefill +; decode 4 ms a round back |
| Sequence-parallel prompt chunks, routed prompt experts for any K, per-shape prompt GEMM tiles, 8192-row chunks for long prompts | prefill 705 → ~1,040-1,120 tok/s at 32K, 128K TTFT 262 → 131 s |
| Concurrent streams (`--parallel N`) with MTP or DFlash2 drafts per stream; short prompts fill whole | 4 streams ~70-95 tok/s aggregate, TTFT ≤ 1.4 s |
| The runner refuses a `--context` × `--parallel` whose caches would not fit | GB10's unified memory swaps instead of failing: 4 × 140K rebooted all four nodes once |

## Node notes that cost us time

- `vm.compaction_proactiveness=0` on every node (compaction stalls a rank ~130 ms at a time).
- RoCE GID indices can move across reboots (an IPv6 address on the fabric NIC took one's slot): resolve the RoCE v2
  GID of each node's fabric IPv4 at launch. After a full-cluster reboot, ping the fabric peers before the first boot.
- Anything else on a rank — a desktop browser, other containers — shows up as rank skew: all four wait for the
  slowest at every all-reduce. Keep the Sparks clean.
- `--parallel 4` needs a smaller context (each stream holds its own cache): `--context 32768` fits.

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
