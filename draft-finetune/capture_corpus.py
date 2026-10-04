#!/usr/bin/env python3
"""On-policy capture: send prompts (build_prompts.py) to a TensorFold server started with TF_GLM53_CAPTURE_DIR, one at a
time (capture runs on the single-stream path). Rank 0 writes one seq-*.safetensors per request: the target's tap rows
and its top-32 soft labels for every token it sampled - exactly what dflash2_train.py trains on. Resumable.

usage: capture_corpus.py --url http://spark1:8890 --prompts prompts.jsonl [--prose 360 --essay 90 --code 200 --list 110 --reason 140]
(the defaults are the 900-prompt mix of our fine-tune; ~9 h at ~27 tok/s, ~29 GB of capture files)
"""
import argparse, json, random, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--prompts", default="prompts.jsonl")
ap.add_argument("--log", default="capture_log.jsonl")
ap.add_argument("--seed", type=int, default=7)
for cls, n in (("prose", 360), ("essay", 90), ("code", 200), ("list", 110), ("reason", 140)):
    ap.add_argument(f"--{cls}", type=int, default=n)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.prompts)]
rng = random.Random(a.seed)
pick = []
for cls in ("prose", "essay", "code", "list", "reason"):
    pool = [r for r in rows if r["cls"] == cls]
    pick += rng.sample(pool, min(getattr(a, cls), len(pool)))
rng.shuffle(pick)
try:
    done = {json.loads(l)["id"] for l in open(a.log)}
except OSError:
    done = set()
t0, toks = time.time(), 0
for i, r in enumerate(pick):
    if r["id"] in done:
        continue
    body = {"model": "glm-5.3-tf", "messages": [{"role": "user", "content": r["text"]}],
            "max_tokens": min(r["max_tokens"], 1024), "temperature": 0.6, "top_p": 0.95, "seed": 1000 + r["id"],
            "chat_template_kwargs": {"enable_thinking": bool(r["think"])}, "tf_mtp": "auto"}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(a.url + "/v1/chat/completions",
                      json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
        n = d.get("usage", {}).get("completion_tokens", 0)
        toks += n
        rec = {"id": r["id"], "cls": r["cls"], "completion_tokens": n, "t": time.time()}
    except Exception as e:  # noqa: BLE001
        rec = {"id": r["id"], "cls": r["cls"], "error": repr(e)[:200], "t": time.time()}
    with open(a.log, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if i % 25 == 0:
        el = time.time() - t0
        print(f"{i + 1}/{len(pick)} prompts, {toks} tokens, {toks / max(el, 1):.1f} tok/s, {el / 3600:.2f} h", flush=True)
print(f"DONE {len(pick)} prompts, {toks} tokens, {(time.time() - t0) / 3600:.2f} h", flush=True)
