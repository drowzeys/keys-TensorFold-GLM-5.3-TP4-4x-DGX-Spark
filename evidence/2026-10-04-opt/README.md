# 2026-10-04-opt (TensorFold branch glm53-tp4-opt)

- `F.log`, `F-h2h.jsonl`: single-stream benches of the final configuration on clean nodes (README tables).
- `C-normed.log`, `C-dflash.log`: 1 / 2 / 4 concurrent streams (`--parallel 4 --context 32768`), MTP and DFlash2
  drafts, greedy and sampled, each reply checked against the same request alone.
- `ab-BASE.log` .. `ab-OPT3.log`: the same-afternoon A/B (BASE = image `2026-10-04`, OPT1 = new code one rail,
  OPT2 = + dual rails, OPT3 = MTP-3); that afternoon the nodes were memory-pressured, so absolute numbers read ~10 %
  low against the clean-node run.
- `R1.log` (4 NCCL channels: rejected, prefill 32K 733 tok/s), `R2.log` + `reuse.log` (prompt-state reuse on: kept
  opt-in, see the README).
- `clocks-under-load.txt`: every rank's GPU clock, power and temperature during benches (all ~2.4 GHz, no cap).
- `tests.summary`: the CUDA test files, four ranks as threads on one GPU each. Notes:
  - first-round failures were fixed and re-run (`opt2`: multi 8 passed, multi_dflash 3 passed): thread-local graph
    capture; the sharded sampler restricted to top_k > 0;
  - `test_exl3_experts.py`'s one failure predates this work: the image's ExLlamaV3 reference build rejects a
    half-integer bit rate in its own `reconstruct()`;
  - `test_glm_moe_dsa_engine.py` (four ranks as threads in one process) failed on a CUDA module loaded during another
    thread's capture, then hung with eager loading; unresolved. It is test-only: the served engine runs one process a
    rank, and every TP4 boot and bench above (12 boots) ran this code.

## Final image (rev 13a79f6 = branch glm53-tp4-opt @ fcf1f20, the same tree; commit authors rewritten), thinking on

- `oneshot-bench.log`: `./one-shot.sh bench` on the published image (prose 41.8, code 38.1 tok/s; 25K prompt 25.0 s,
  needle PASS) - the README summary.
- `final-trace-bench.log`, `final-memory-trace.log`, `final-rank0.log`: single-stream benches (MTP / DFlash2, short /
  32K, 128K prefill) with every node's MemAvailable / swap every ~6 s: 15 GB idle, 9 GB minimum under 32K / 128K
  prompts, swap 0 throughout (`vm.swappiness=1`, page cache dropped at start and during load).
- Memory history: the first published-path smoke test crashed (GPU out of memory) and later boots filled 15 GB of swap
  while loading; the cause was the checkpoint's page cache (above all on the NFS-exporting node) plus ~10 GB of extra
  decode graphs from finer index buckets. Fixed in the engine, node settings and one-shot.sh.
