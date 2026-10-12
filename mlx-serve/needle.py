#!/usr/bin/env python3
"""needle.py URL LINES: a passphrase in the middle of LINES filler lines (~21 tokens a line); the model must return it."""
import json, sys, time, random, urllib.request
url, n = sys.argv[1], int(sys.argv[2]); rnd = random.Random(7)
goods = ["copper fittings", "glass jars", "wool blankets", "steel bolts", "paper reams", "ceramic tiles", "oak planks", "rubber seals"]
lines = [f"Entry {i:06d}: the warehouse in sector {rnd.randint(1, 99)} holds {rnd.randint(100, 999)} crates of {rnd.choice(goods)}." for i in range(n)]
secret = "quartz-heron-4417"
lines.insert(n // 2, f"Note for the auditor: the vault passphrase is {secret}.")
q = "\n".join(lines) + "\n\nWhat is the vault passphrase in the note for the auditor? Answer with the passphrase only."
body = {"model": "model", "messages": [{"role": "user", "content": q}], "max_tokens": 300, "temperature": 0}
t0 = time.time()
d = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"}), timeout=3600))
dt = time.time() - t0; ans = d["choices"][0]["message"]["content"] or ""; u = d.get("usage", {}); t = d.get("timings", {})
print(f"needle {n} lines: prompt {u.get('prompt_tokens')} tokens, {dt:.1f} s, prefill {t.get('prompt_per_second', 0):.0f} tok/s, decode {t.get('predicted_per_second', 0):.1f} tok/s, "
      f"{'FOUND' if secret in ans else 'MISSED'} answer={ans.strip()[:60]!r}")
