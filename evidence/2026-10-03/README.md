# 2026-10-03 runs (four DGX Sparks, full GLM-5.3 2.75 bpw EXL3, TensorFold `final-all`)

| File | What | Draft |
|---|---|---|
| `tf_public_bench.summary` | **the README's decode + concurrency numbers** (3 repeats; `--alone` checks) | public incoai/GLM-5.3-DFlash2 |
| `tf_full_suite.summary` | full suite: decode 32K + short, prefill/TTFT 4K-128K, needles, concurrency, node health | our on-policy fine-tune (not redistributed); prefill/MTP rows are draft-independent |
| `tf_nccl_pf.summary` | prefill with NCCL 4 channels / LL128 vs the default (Simple, 2 channels): default wins | - |
| `tf_bisect_det.summary` | run-to-run reproducibility bisection: multi-chunk prompts diverge with the atomic MoE prompt kernel, under every comm setting | - |
| `tf_fix_validate.summary` | reproducible prompt-expert sums: unit tests, e2e identity at 8K/32K, prefill cost of `slots`, `fixed`, `slots16` | - |

Bisection columns: three identical greedy requests after a warm-up; each pair's first differing token ('=': identical).
