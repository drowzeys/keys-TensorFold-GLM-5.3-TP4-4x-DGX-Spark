#!/usr/bin/env python3
"""Checks of two streams at once (PRESET=250k). URL=http://<rank 0>:18992. Thinking on (the server's default), greedy.

  streams.py replies OUT.json --streams S      the fixed prompts, S at a time (S 1: one after another); keeps every reply
  streams.py compare BASE.json OTHER.json      how many replies (reasoning + content) are identical
  streams.py needle OUT.json --streams S --lines N    S needles at once, a different passphrase each; FOUND / MISSED each
  streams.py bench OUT.json --streams S        prose and code, 512 tokens: 1, 2 .. S requests at once; per-stream and
                                               aggregate tok/s (completion tokens over the wall time of the whole set)
"""
import argparse, json, random, sys, threading, time, urllib.request

import os
URL = os.environ.get("URL", "http://127.0.0.1:18992") + "/v1/chat/completions"   # URL=http://<rank 0>:18992
GOODS = ["copper fittings", "glass jars", "wool blankets", "steel bolts", "paper reams", "ceramic tiles", "oak planks", "rubber seals"]


def inventory(n, seed, secret=None):
    r = random.Random(seed)
    rows = [f"Entry {i:06d}: the warehouse in sector {r.randint(1, 99)} holds {r.randint(100, 999)} crates of {r.choice(GOODS)}." for i in range(n)]
    if secret:
        rows.insert(n // 2, f"Note for the auditor: the vault passphrase is {secret}.")
    return "\n".join(rows)


# short ones, one that crosses the selection cut (2,048 tokens) while it decodes, long ones in different key buckets
FIXED = [
    ("prose", "Write a 400-word essay on why the ocean is salty."),
    ("code", "Write a Python class implementing an LRU cache with get and put, with type hints and docstrings, then a few unit tests."),
    ("edge", inventory(93, 11) + "\n\nList the three sectors with the most crates of oak planks, with your working."),
    ("long9k", inventory(420, 12) + "\n\nSummarise this inventory in five sentences, then write a Python function that parses one of its lines."),
    ("long19k", inventory(900, 13) + "\n\nWrite a limerick about this inventory, then name the entry with the largest crate count you can find."),
    ("story", "Write a short story about a clockmaker who repairs a clock that runs backwards."),
    ("rust", "Write a Rust function that parses a CSV line with quoted fields, with tests."),
    ("long5k", inventory(240, 14) + "\n\nTranslate the first three entries into French, then write a short poem about warehouses."),
]
BENCH = {"prose": ["Write a 400-word essay on why the ocean is salty.", "Explain how vaccines train the immune system, for a curious teenager.",
                   "Describe the causes and consequences of the fall of the Western Roman Empire.", "Write a short story about a clockmaker who repairs a clock that runs backwards."],
         "code": ["Write a Python class implementing an LRU cache with get and put, with type hints and docstrings, then a few unit tests.",
                  "Write a Rust function that parses a CSV line with quoted fields, with tests.", "Implement Dijkstra's shortest path in Python with a heap, and explain its complexity.",
                  "Write a TypeScript debounce function with generics and a usage example."]}


def call(content, n, timeout=7200):
    body = {"model": "model", "messages": [{"role": "user", "content": content}], "max_tokens": n, "temperature": 0}
    t = time.time()
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"}), timeout=timeout))
    except Exception as e:  # noqa: BLE001 - a failed stream is a result
        return {"error": str(e)[:300], "secs": round(time.time() - t, 2), "content": "", "reasoning": "", "usage": {}, "timings": {}}
    m = d["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "", "finish": d["choices"][0].get("finish_reason"),
            "timings": d.get("timings") or {}, "usage": d.get("usage") or {}, "secs": round(time.time() - t, 2), "error": None}


def together(jobs):
    """[(content, max_tokens)] sent at once -> their replies in order, and the wall time of the whole set."""
    out = [None] * len(jobs)

    def run(i):
        out[i] = call(*jobs[i])
    ts = [threading.Thread(target=run, args=(i,)) for i in range(len(jobs))]
    t = time.time()
    for x in ts:
        x.start()
    for x in ts:
        x.join()
    return out, time.time() - t


def replies(a):
    res = {}
    for i in range(0, len(FIXED), a.streams):
        part = FIXED[i:i + a.streams]
        got, wall = together([(p, 160) for _, p in part])
        for (name, _), r in zip(part, got):
            res[name] = r
            print(f"{name}: prompt {r['usage'].get('prompt_tokens')} tokens, {r['usage'].get('completion_tokens')} out, {r['secs']} s{' ERROR ' + r['error'] if r['error'] else ''}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    return 1 if any(r["error"] for r in res.values()) else 0


def compare(a):
    x, y = json.load(open(a.out)), json.load(open(a.other))
    same = [k for k in x if k in y and not x[k]["error"] and not y[k]["error"] and x[k]["content"] == y[k]["content"] and x[k]["reasoning"] == y[k]["reasoning"]]
    diff = [k for k in x if k not in same]
    print(f"{len(same)} of {len(x)} replies identical" + (f"; differ: {', '.join(diff)}" if diff else ""))
    return 0 if not diff else 1


def needle(a):
    secrets = [f"{w}-{1000 + 37 * i}" for i, w in enumerate(["quartz-heron", "amber-otter", "cobalt-finch", "garnet-lynx"])][:a.streams]
    jobs = [(inventory(a.lines, 70 + i, s) + "\n\nWhat is the vault passphrase in the note for the auditor? Answer with the passphrase only.", 300) for i, s in enumerate(secrets)]
    got, wall = together(jobs)
    ok = True
    for i, (r, s) in enumerate(zip(got, secrets)):
        found = s in r["content"]
        ok = ok and found
        t = r["timings"]
        print(f"needle stream {i}: prompt {r['usage'].get('prompt_tokens')} tokens, {r['secs']} s, prefill {t.get('prompt_per_second', 0):.0f} tok/s, "
              f"decode {t.get('predicted_per_second', 0):.1f} tok/s, {'FOUND' if found else 'MISSED'} answer={r['content'].strip()[:50]!r}{' ERROR ' + r['error'] if r['error'] else ''}", flush=True)
    tokens = sum(r["usage"].get("prompt_tokens") or 0 for r in got)
    print(f"needle {a.streams} streams x {a.lines} lines: {tokens} prompt tokens in {wall:.0f} s = {tokens / wall:.0f} tok/s in all; {'ALL FOUND' if ok else 'SOME MISSED'}")
    json.dump(got, open(a.out, "w"), indent=1)
    return 0 if ok else 1


def bench(a):
    res = {}
    counts = [n for n in (1, 2, 4, 8) if n <= a.streams]
    for kind, ps in BENCH.items():
        for n in counts:
            best = None
            for _ in range(2):
                got, wall = together([(ps[i % len(ps)], 512) for i in range(n)])
                toks = sum(r["usage"].get("completion_tokens") or 0 for r in got)
                per = [r["timings"].get("predicted_per_second") or 0 for r in got]
                run = {"aggregate": toks / wall, "wall": wall, "tokens": toks, "per_stream": per, "errors": [r["error"] for r in got if r["error"]]}
                if best is None or run["aggregate"] > best["aggregate"]:
                    best = run
            res[f"{kind}|{n}"] = best
            print(f"{kind} x {n}: aggregate {best['aggregate']:.1f} tok/s ({best['tokens']} tokens in {best['wall']:.1f} s); per stream "
                  f"{', '.join(f'{x:.1f}' for x in best['per_stream'])}{' ERRORS ' + str(best['errors']) if best['errors'] else ''}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["replies", "compare", "needle", "bench"])
    ap.add_argument("out")
    ap.add_argument("other", nargs="?")
    ap.add_argument("--streams", type=int, default=1)
    ap.add_argument("--lines", type=int, default=2800)
    a = ap.parse_args()
    return {"replies": replies, "compare": compare, "needle": needle, "bench": bench}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
