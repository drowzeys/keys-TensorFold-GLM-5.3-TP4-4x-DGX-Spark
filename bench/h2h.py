"""bertholomus' bench/tf_greedy.py prompts (greedy, thinking on (the server default; theirs ran thinking off), 300 tokens, timed like theirs: completion tokens over
the whole request) against our server, per draft mode. usage: h2h.py LABEL MODE [REPEATS] [URL]"""
import json, sys, time, urllib.request
U = sys.argv[4] if len(sys.argv) > 4 else "http://127.0.0.1:8890"
P = ["Write a detailed explanation of how a hash table works, including collisions, load factor and resizing.",
     "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
     "List the planets of the solar system with one interesting fact about each.",
     "Translate to French: The weather is nice today, so we will go for a walk in the park after lunch."]
label, mode, reps = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3
for p in P:
    rates, tpr = [], []
    for _ in range(reps):
        body = {"model": "glm-5.3-tf", "messages": [{"role": "user", "content": p}], "max_tokens": 300, "temperature": 0,
                "tf_mtp": mode}
        t = time.time()
        b = json.load(urllib.request.urlopen(urllib.request.Request(U + "/v1/chat/completions", json.dumps(body).encode(),
                                             {"Content-Type": "application/json"}), timeout=900))
        dt = time.time() - t; n = b["usage"]["completion_tokens"]; rates.append(n / dt)
        tpr.append((b.get("tensorfold") or {}).get("tokens_per_round"))
    print(json.dumps({"label": label, "mode": mode, "prompt": p[:28], "tokens": n, "client_tps": round(sum(rates) / len(rates), 2),
                      "runs": [round(r, 2) for r in rates], "tokens_per_round": tpr[-1]}), flush=True)
