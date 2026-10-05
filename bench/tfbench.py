#!/usr/bin/env python3
"""bench.py's single-stream decode bench for the TensorFold server (same prompts, 32K context file, sampling, 512
tokens); reads the engine's own stats (tokens per round, accept) from the final stream chunk instead of /metrics.

usage: tfbench.py LABEL [--base URL] [--mode normed/raw] [--repeats 3] [--tasks a,b] [--no-context] [--greedy]
"""
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

from bench import PROMPTS


def run(a, task, seed, max_tokens=None):
    prompt = PROMPTS[task]
    if not a.no_context:
        prompt = "Background reading:\n" + Path(a.context_file).read_text() + "\n\nTask:\n" + prompt
    body = dict(model="glm-5.3-tf", messages=[dict(role="user", content=prompt)], max_tokens=max_tokens or a.tokens,
                stream=True, seed=seed,
                temperature=0.0 if a.greedy else a.temperature, top_p=0.95)
    if a.mode:
        body["tf_mtp"] = a.mode
    if a.serial:
        body["draft"] = False
    req = urllib.request.Request(a.base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    start, first, last, n, stats = time.perf_counter(), None, None, 0, {}
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            stats = ev.get("tensorfold") or stats
            ch = ev.get("choices") or []
            if ch and ((ch[0].get("delta") or {}).get("content") or (ch[0].get("delta") or {}).get("reasoning_content")):
                now = time.perf_counter()
                first = first or now
                last = now
                n += 1
    tok = stats.get("tokens") or n
    dec = stats.get("decode_s") or ((last - first) if first and last else 0)
    return dict(label=a.label, task=task, seed=seed, mode=stats.get("mtp_mode"), ttft_s=(first or start) - start,
                prefill_s=stats.get("prefill_s"), tokens=tok, decode_tps=(tok - 1) / dec if dec else 0,
                tokens_per_round=stats.get("tokens_per_round"), accept=stats.get("accept"),
                pass_ms=1000 * dec / stats["rounds"] if stats.get("rounds") else None, graphs=stats.get("graphs"),
                capture_s=stats.get("capture_s"), sha=stats.get("sha256"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("label")
    p.add_argument("--base", default="http://localhost:8890")
    p.add_argument("--mode")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--tasks", default=",".join(PROMPTS))
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--context-file", default="context-32k.txt")
    p.add_argument("--no-context", action="store_true")
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--serial", action="store_true")
    p.add_argument("--out", default="tfbench.jsonl")
    a = p.parse_args()
    w = run(a, "prose_beekeeper", 999, max_tokens=48)
    print(f"warmup: {w['decode_tps']:.2f} tok/s, ttft {w['ttft_s']:.1f}s", flush=True)
    rows = []
    with open(a.out, "a") as f:
        for rep in range(a.repeats):
            for t in a.tasks.split(","):
                r = run(a, t, 1729 + rep)
                rows.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                print(f"{t:18s} {r['decode_tps']:6.2f} tok/s  tok/round {r['tokens_per_round']}  pass "
                      f"{r['pass_ms'] and round(r['pass_ms'], 1)} ms  ttft {r['ttft_s']:.1f}s", flush=True)
    for kind in ("prose", "code"):
        v = [r["decode_tps"] for r in rows if r["task"].startswith(kind)]
        tpr = [r["tokens_per_round"] for r in rows if r["task"].startswith(kind) and r["tokens_per_round"]]
        pm = [r["pass_ms"] for r in rows if r["task"].startswith(kind) and r["pass_ms"]]
        if v:
            print(f"{a.label:24s} {kind:6s} n={len(v)} tps {statistics.median(v):.2f} tok/round "
                  f"{statistics.median(tpr) if tpr else 0:.2f} pass {statistics.median(pm) if pm else 0:.1f} ms")


if __name__ == "__main__":
    main()
