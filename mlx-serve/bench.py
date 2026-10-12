#!/usr/bin/env python3
"""Decode speed of a running server: one prose and one code prompt, greedy, the default thinking mode.
   bench.py [URL] [max_tokens=512] [extra request JSON]"""
import json, sys, urllib.request
PROMPTS = {
    "prose": "Write a 400-word essay on why the ocean is salty.",
    "code": "Write a Python class implementing an LRU cache with get and put, with type hints and docstrings, then a few unit tests.",
}
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18992"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 512
extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
for name, p in PROMPTS.items():
    rates = []
    for rep in range(2):
        body = {"model": "model", "messages": [{"role": "user", "content": p}], "max_tokens": N, "temperature": 0, **extra}
        r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
        d = json.load(urllib.request.urlopen(r, timeout=600))
        t = d["timings"]; rates.append(t["predicted_per_second"]); txt = d["choices"][0]["message"]["content"] or ""
    print(f"{name}: {max(rates):.1f} tok/s (runs {', '.join(f'{x:.1f}' for x in rates)}; {t['predicted_n']} tokens; text {__import__('hashlib').md5(txt.encode()).hexdigest()[:8]})")
