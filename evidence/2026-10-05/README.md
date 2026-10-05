# 2026-10-05

- `oneshot-bench.log`: `./one-shot.sh bench` on image `2026-10-05` (prompt reuse on): prose 40.2 / code 39.0 tok/s,
  25K prompt 25.3 s, needle PASS.
- `reuse-1stream.log`: `tools/glm53_reuse_check.py --ctx 50000 --system 8192` single stream (turn 2 at 93K: 85.7 s cold,
  0.85 s warm; resend 0.15 s; shared 14.7K system prompt 0.38 s).
- `reuse-4stream.log`, `conc-4stream.jsonl`, `memtrace-4stream.log`: `PARALLEL=4 CONTEXT=32768`: reuse check (14K / 6K),
  `bench/conc_chat.py` 1 and 4 streams (78.7 prose / 100.2 code aggregate), memory every 5 s (minimum 11 GB, no swap).
- `needle.jsonl`, `requests.txt`, `memtrace.log`: the 1M validation (image `2026-10-04-opt`, `CONTEXT=1000000`):
  needles at 115K / 456K / 905K all PASS; prefill 394 / 352 / 307 tok/s; decode 37.0 / 35.7 / 33.7 tok/s; 15-16 GB free.
- Memory incident behind the `up` wait: a 4-stream lane started seconds after a single-stream lane stopped dropped to
  ~1 GB free (guard stopped it); the same image and requests started clean held 11-15 GB.
