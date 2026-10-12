#!/usr/bin/env python3
"""Hang / early-exit / loop checks against the running server. Every step has a deadline.
   HOST=<rank 0 address> PORT=18992 checks.py [stop long think sampled drop dropmany title over tool longprompt]"""
import json, os, socket, sys, threading, time, urllib.request, collections
HOST = os.environ.get("HOST", "127.0.0.1"); PORT = int(os.environ.get("PORT", "18992"))
URL = f"http://{HOST}:{PORT}/v1/chat/completions"
def call(content, max_tokens, timeout=600, **extra):
    body = {"model": "model", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0, **extra}
    r = urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"})
    t = time.time()
    try:
        d = json.load(urllib.request.urlopen(r, timeout=timeout))
    except Exception as e:
        return {"error": repr(e), "secs": time.time() - t}
    c = d["choices"][0]; m = c["message"]
    return {"finish": c.get("finish_reason"), "n": d["usage"]["completion_tokens"], "secs": round(time.time() - t, 1),
            "text": m.get("content") or "", "think": m.get("reasoning_content") or "", "tools": m.get("tool_calls")}
def loopiness(text):
    w = text.split()
    if len(w) < 40: return 0.0
    g = collections.Counter(tuple(w[i:i+8]) for i in range(len(w) - 7))
    return round(sum(c - 1 for c in g.values() if c > 1) / max(1, len(w) - 7), 3)
def show(name, r):
    if "error" in r: print(f"FAIL {name}: {r}", flush=True); return
    print(f"{name}: finish={r['finish']} tokens={r['n']} {r['secs']}s loop={loopiness(r['text'] + ' ' + r['think'])} content={len(r['text'])}ch think={len(r['think'])}ch tail={r['text'][-70:]!r}", flush=True)
def drop(content, after_s, max_tokens=2000):
    """Start a streamed reply and close the socket abruptly after `after_s`."""
    body = json.dumps({"model": "model", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens, "temperature": 0, "stream": True}).encode()
    s = socket.create_connection((HOST, PORT), timeout=30)
    s.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
    time.sleep(after_s); got = 0
    s.setblocking(False)
    try: got = len(s.recv(1 << 20))
    except Exception: pass
    s.close(); return got
def par(jobs):
    out = [None] * len(jobs)
    def run(i, f): out[i] = f()
    th = [threading.Thread(target=run, args=(i, f)) for i, f in enumerate(jobs)]
    [t.start() for t in th]; [t.join() for t in th]; return out

which = sys.argv[1:] or ["stop", "long", "think", "sampled", "drop", "dropmany", "title", "over", "tool", "longprompt"]
for w in which:
    print(f"--- {w}", flush=True)
    if w == "stop":
        show("short answer", call("What is the capital of France? One word.", 2000))
        show("medium answer", call("Explain in three sentences how a refrigerator works.", 2000))
        show("max_tokens=1", call("Say hello.", 1))
    elif w == "long":
        show("essay 6000", call("Write a detailed 2500-word technical guide to B-trees: structure, search, insert, delete, with pseudocode.", 6000))
    elif w == "think":
        show("thinking on", call("A farmer has 17 sheep, all but 9 run away. How many are left? Then explain.", 4000, enable_thinking=True))
        show("thinking code", call("Write a Python function that merges overlapping intervals, with tests.", 4000, enable_thinking=True))
    elif w == "sampled":
        for s in (1, 2):
            show(f"temp 0.8 seed {s}", call("Write a 300-word story about a lighthouse keeper.", 1500, temperature=0.8, top_p=0.95, seed=s))
    elif w == "drop":
        for after in (0.05, 0.5, 3.0):
            got = drop("Write a 1000-word essay on volcanoes.", after)
            t = time.time(); r = call("Reply with the single word: ok", 50, timeout=120)
            print(f"dropped after {after}s ({got} bytes seen); next request: {r.get('finish', r.get('error'))} in {round(time.time() - t, 1)}s", flush=True)
    elif w == "dropmany":
        def d(i): return lambda: drop(f"Write a 1000-word essay on topic number {i}.", 1.0 + i * 0.4)
        def k(i): return lambda: call(f"Write 150 words about the number {i}.", 400, timeout=300)
        res = par([d(0), k(1), d(2), k(3)])
        for i in (1, 3): show(f"survivor {i}", res[i])
        res = par([k(i) for i in range(4)])
        for i, r in enumerate(res): show(f"after drops, 4 at once #{i}", r)
    elif w == "title":
        # the pattern that deadlocked the engine's Turns server: a reply streaming, a short title request beside it, then a disconnect
        res = par([lambda: drop("Write a 1000-word essay on glaciers.", 2.0), lambda: call("Give a 4-word title for a chat about glaciers.", 30, timeout=120)])
        show("title beside dropped stream", res[1])
        show("request after", call("Reply with the single word: ok", 50, timeout=120))
    elif w == "over":
        res = par([(lambda i=i: call(f"Write 120 words about the colour number {i}.", 300, timeout=400)) for i in range(7)])
        for i, r in enumerate(res): show(f"7 at once (4 lanes) #{i}", r)
    elif w == "tool":
        tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        r = call("What's the weather in Paris right now?", 500, tools=tools)
        show("tool call", r); print("  tool_calls:", json.dumps(r.get("tools"))[:300], flush=True)
    elif w == "longprompt":
        doc = " ".join(f"Entry {i}: the warehouse in sector {i % 37} holds {i * 7 % 911} crates." for i in range(1400))
        show("~25K prompt", call(doc + "\n\nHow many crates does entry 1203 hold? Answer with the number only.", 200, timeout=900))
        show("same prompt again", call(doc + "\n\nHow many crates does entry 77 hold? Answer with the number only.", 200, timeout=900))
        show("small after long", call("Reply with the single word: ok", 50, timeout=120))
print("--- done", flush=True)
