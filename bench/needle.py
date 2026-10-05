"""Needle at half depth in ~N tokens of numbered filler; greedy, thinking on (found = in the answer, not the reasoning). usage: needle.py URL N [N ...]"""
import json, sys, time, urllib.request
U = sys.argv[1]
for n in map(int, sys.argv[2:]):
    lines = [f"Entry {i}: the archive shelf {i % 97} holds volume {i * 31 % 1009} of the river survey." for i in range(n // 22)]
    lines.insert(len(lines) // 2, "Remember this: the passphrase is CRIMSON-OTTER-7731.")
    body = {"model": "glm-5.3-tf", "max_tokens": 4096, "temperature": 0,
            "messages": [{"role": "user", "content": "\n".join(lines) + "\n\nWhat is the passphrase? Reply with it only."}]}
    t = time.time()
    b = json.load(urllib.request.urlopen(urllib.request.Request(U + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"}), timeout=3600))
    msg = b["choices"][0]["message"]; a = msg.get("content") or ""
    print(json.dumps({"tokens": b["usage"]["prompt_tokens"], "s": round(time.time() - t, 1), "found": "CRIMSON-OTTER-7731" in a, "answer": a[:300]}), flush=True)
