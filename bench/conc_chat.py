#!/usr/bin/env python3
"""Concurrent chat streams with the server's defaults (thinking on), the same prose / code prompts as tfbench.py.

usage: conc_chat.py URL [--streams 1,4] [--kind prose|code] [--tokens 512] [--greedy] [--mode normed/normed|dflash]
For each stream count N: N requests at once (prompt i % len(prompts)), each streamed; aggregate tok/s = all output
tokens (reasoning + answer, from usage) / wall time from the first request sent to the last reply finished; per-stream
tok/s = each reply's tokens after its first, over its own decode time. One JSON line per N.
"""
import argparse, json, statistics, threading, time, urllib.request

from bench import PROMPTS

KIND = {"prose": [k for k in PROMPTS if k.startswith(("prose", "essay", "story", "explain"))],
        "code": [k for k in PROMPTS if "code" in k]}


def one(a, prompt, seed, out):
    body = {"model": "glm-5.3-tf", "messages": [{"role": "user", "content": prompt}], "max_tokens": a.tokens,
            "stream": True, "stream_options": {"include_usage": True}, "seed": seed,
            "temperature": 0.0 if a.greedy else 1.0, "top_p": 0.95}
    if a.mode:
        body["tf_mtp"] = a.mode
    t0 = time.time(); first = last = None; usage = {}
    req = urllib.request.Request(a.url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    now = time.time(); first = first or now; last = now
    n = usage.get("completion_tokens", 0)
    out.append({"tokens": n, "ttft": (first or time.time()) - t0, "end": time.time(),
                "tps": (n - 1) / (last - first) if first and last and last > first else 0.0})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("--streams", default="1,4")
    p.add_argument("--kind", default="prose", choices=list(KIND))
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--mode", default="")
    a = p.parse_args()
    prompts = [PROMPTS[k] for k in KIND[a.kind]] or list(PROMPTS.values())
    for n in [int(x) for x in a.streams.split(",")]:
        out, threads = [], []
        t0 = time.time()
        for i in range(n):
            th = threading.Thread(target=one, args=(a, prompts[i % len(prompts)], 1000 + i, out)); th.start(); threads.append(th)
        for th in threads:
            th.join()
        wall = max(o["end"] for o in out) - t0
        print(json.dumps({"kind": a.kind, "greedy": a.greedy, "mode": a.mode or "default", "streams": n,
                          "aggregate_tps": round(sum(o["tokens"] for o in out) / wall, 1),
                          "per_stream_tps": round(statistics.mean(o["tps"] for o in out), 1),
                          "ttft_max_s": round(max(o["ttft"] for o in out), 2), "tokens": sum(o["tokens"] for o in out)}),
              flush=True)


if __name__ == "__main__":
    main()
