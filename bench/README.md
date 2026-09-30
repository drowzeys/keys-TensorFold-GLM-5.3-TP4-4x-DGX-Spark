# Benches

Run from a directory holding `context-32k.txt`: any ~140 KB of plain text of your choice (the "background reading"
that puts every request at ~32K tokens of context; not included here).

- `tfbench.py LABEL [--mode normed/normed|dflash|auto] [--no-context] [--greedy]` — single-stream decode against a
  TensorFold server (bench.py's prompts: 3 prose + 2 code, temperature 1.0, top-p 0.95, 512 tokens); prints tok/s,
  tokens per verify step and step time from the engine's own stats.
- `bench.py` — the same bench against a vLLM server (reads vLLM's /metrics for the MTP acceptance).
- `ppbench.py LABEL URL MODEL --lengths 4000,8000,32000` — prefill tok/s and TTFT at fresh prompts of each length.
