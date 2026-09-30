# Evidence for ashhart/TensorFold#159

## Versions (SHAs)

| What | Version |
|---|---|
| TensorFold base | v0.5.0 `9cd52ab` |
| PR branch `drowzeys/TensorFold:glm-moe-dsa-tp4` | `cfd93db` (TP=4 engine) + `dc43ca6` (1M DCP, DFlash2 ring, capture) |
| Hardware | 4 × NVIDIA DGX Spark (GB10, 128 GB unified), ConnectX-7 200 GbE RoCE (one port each), driver 580.173.02, kernel 6.17.0-1029-nvidia |
| Software in the serving container | CUDA 13.0, torch 2.13.0+cu130, Triton 3.7.1, NCCL 2.30.7; optional: cuda-exl3 (native), b12x `b58f34e` (RoCEnante) |
| Target checkpoint | [drowzeys/keys-GLM-5.3-EXL3-2.75BPW](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW) (repo head `b0328d0`) |
| DFlash2 draft | [incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) repo `425aa615`, `model.safetensors` sha256 `3105f140…` |
| Prebuilt image | `ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-09-30` (sha256 `a1f9ce24…`; predates `dc43ca6`) |

## Logs and results

- `logs/rank{0..3}-serve.log` — the four ranks of a serve (36K context, DFlash2 drafter loaded): the RoCE one-shot check
  (`ranks bit-equal: True; equals the NCCL rank-order sum: True`), linear tiling, memory, graph capture. Fabric
  addresses and paths redacted.
- `results/tensorfold-decode-bench.jsonl` — every TensorFold decode bench row (labels: `tf-final-32k` MTP,
  `tf-dflash-32k`, `tf-auto-32k`, `ab-*` / `abp-*` MTP input variants; `tokens_per_round`, `pass_ms`, `sha`).
- `results/vllm-decode-bench.jsonl` — the vLLM rows on the same checkpoint and hardware (same prompts and sampling).
- `results/prefill-ttft.jsonl` — prefill / TTFT (TensorFold `tf-*`, vLLM `k35-roce-nocompact`).
- `results/needles.jsonl`, `results/1m-needles-summary.txt` — the 1M-context (DCP 4) needles (the first ~1M attempt was
  rejected at 1,029K tokens — past the 1M limit — and rerun at 966,562).
- `results/mtp-input-variants-code.log` — the MTP input A/B behind the `normed/normed` default.

## Checking the rank protocol with fewer GPUs

**One Spark, today:** `tests/cuda/test_glm_moe_dsa_{model,fused}.py` run four ranks as threads of one process on one
GPU (`tests/cuda/threadcomm.py`: a `comm.NCCL` stand-in with the same all-gather / all-to-all semantics), on the real
checkpoint's layers 0-3 (+ the MTP layer, + the DFlash2 draft): the four-way split, rank-order sums, a 3-row verify
window == 3 serial steps, rows past index_topk, wide prompt chunks, DCP 4, and MTP / DFlash2 / auto replies == serial.

```sh
TF_GLM53_CKPT=/models/GLM-5.3-EXL3-2.75BPW TF_GLM53_DFLASH_TEST=/models/GLM-5.3-DFlash2 \
  python -m pytest -v tests/test_glm_moe_dsa_split.py tests/cuda/test_glm_moe_dsa_fused.py
```

(~40 GB of the one GPU: four ranks' shares of four layers, the embedding and head.)

**Four processes on two Sparks:** NCCL refuses two ranks on one GPU, so this needs a CPU (gloo) communicator for tests
plus a layer subset (four ranks' full shares, ~68 GB each, do not fit in two Sparks). Proposed: `TF_GLM53_COMM=gloo`
+ `--layers N` in the engine, graphs and RoCE off in that mode — slow, but the same rank protocol end to end (HTTP rank
0, followers, header sharing, prompt chunks, MTP and DFlash2 rounds, DCP).
