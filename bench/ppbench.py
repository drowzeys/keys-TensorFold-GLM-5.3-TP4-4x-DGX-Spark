#!/usr/bin/env python3
"""Prefill (PP) + TTFT + decode-at-depth bench for the TP4 serve.

usage: ppbench.py LABEL [URL] [MODEL] [--lengths 1000,4000,16000,32000,64000,128000] [--runs 2] [--decode 128]
Each prompt is fresh (a unique nonce line first + shuffled paragraphs of context-32k.txt), so no prefix reuse.
Per length: TTFT, prefill tok/s (= prompt tokens / TTFT), decode tok/s over the reply (after the first token).
Appends one JSON line per length to ppbench.jsonl.
"""
import argparse, json, random, statistics, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("url", nargs="?", default="http://localhost:8888")
ap.add_argument("model", nargs="?", default=None)
ap.add_argument("--lengths", default="1000,4000,16000,32000,64000,128000")
ap.add_argument("--runs", type=int, default=2)
ap.add_argument("--decode", type=int, default=128)
a = ap.parse_args()
model = a.model or json.load(urllib.request.urlopen(a.url + "/v1/models"))["data"][0]["id"]
paras = [p for p in open("context-32k.txt").read().split("\n") if p.strip()]
CHARS_PER_TOK = 4.3


def prompt(n_tok, seed):
    rng = random.Random(seed)
    out, size = [f"Session {seed}-{time.time_ns()}: read the notes below, then continue the story in the last line."], 0
    while size < n_tok * CHARS_PER_TOK:
        p = rng.choice(paras)
        out.append(p)
        size += len(p) + 1
    out.append("Continue the story from here in plain prose:")
    return "\n".join(out)


def run(n_tok, seed):
    body = {"model": model, "messages": [{"role": "user", "content": prompt(n_tok, seed)}], "max_tokens": a.decode,
            "temperature": 0.6, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); first = last = None; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            usage = d.get("usage") or usage
            for c in d.get("choices", []):
                dl = c.get("delta", {})
                if dl.get("content") or dl.get("reasoning_content"):
                    now = time.time(); first = first or now; last = now
    ttft = first - t0
    ptok, ctok = usage["prompt_tokens"], usage["completion_tokens"]
    dec = (ctok - 1) / (last - first) if last > first and ctok > 1 else 0.0
    return {"prompt_tokens": ptok, "ttft_s": ttft, "prefill_tps": ptok / ttft, "decode_tps": dec, "completion_tokens": ctok}


for n in [int(x) for x in a.lengths.split(",")]:
    rows = [run(n, 1000 * n + i) for i in range(a.runs)]
    best = min(rows, key=lambda r: r["ttft_s"])
    res = {"label": a.label, "target_tokens": n, "runs": rows,
           "ttft_s_median": statistics.median(r["ttft_s"] for r in rows),
           "prefill_tps_median": statistics.median(r["prefill_tps"] for r in rows),
           "decode_tps_median": statistics.median(r["decode_tps"] for r in rows)}
    print(f"{a.label:18s} prompt {best['prompt_tokens']:7d} tok  TTFT {res['ttft_s_median']:7.2f} s  "
          f"prefill {res['prefill_tps_median']:7.0f} tok/s  decode@depth {res['decode_tps_median']:5.1f} tok/s  "
          f"(runs {len(rows)})", flush=True)
    with open("ppbench.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
