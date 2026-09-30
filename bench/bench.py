#!/usr/bin/env python3
"""Seeded, sequential streaming benchmark with explicit sampling and raw output."""
import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path

PROMPTS = {
    "prose_beekeeper": "Write a vivid literary short story (at least 700 words) about a beekeeper during a drought. No headings, flowing prose only.",
    "prose_lighthouse": "Write a vivid literary short story (at least 700 words) about a lighthouse keeper in 1890s Norway. No headings, flowing prose only.",
    "prose_nurse": "Write a vivid literary short story (at least 700 words) about a night-shift nurse in Lagos. No headings, flowing prose only.",
    "code_cache": "Write a complete, well-documented Python module implementing an LRU cache with TTL expiry, thread safety, statistics, and a small CLI demo. Name the class Cache7. Code only, no prose.",
    "code_parser": "Write a complete Python recursive-descent parser for arithmetic expressions with parentheses, unary minus, exponentiation, multiplication and addition. Include helpful syntax errors, examples, and unittest tests. Code only.",
}


def metrics(base):
    result = {}
    with urllib.request.urlopen(base + "/metrics", timeout=20) as response:
        for line in response.read().decode().splitlines():
            if line.startswith("vllm:") and not line.startswith("#"):
                name, _, value = line.rpartition(" ")
                if any(x in name for x in ("spec_decode_num_", "num_requests_running", "num_requests_waiting", "request_success_total")):
                    result[name] = float(value)
    return result


def total(values, name):
    return sum(v for k, v in values.items() if k.startswith("vllm:" + name + "{"))


def run(args, task, seed, max_tokens=None):
    before = metrics(args.base)
    if total(before, "num_requests_running") or total(before, "num_requests_waiting"):
        raise RuntimeError("Endpoint is busy; single-stream benchmark would be contaminated")
    prompt = PROMPTS[task]
    if args.context_file:
        prompt = "Background reading:\n" + Path(args.context_file).read_text() + "\n\nTask:\n" + prompt
    sampling = dict(temperature=args.temperature, top_p=0.95, top_k=-1,
                    repetition_penalty=1.0, seed=seed)
    body = dict(model=args.model, messages=[dict(role="user", content=prompt)],
                max_tokens=max_tokens or args.tokens, stream=True,
                stream_options=dict(include_usage=True),
                chat_template_kwargs=dict(enable_thinking=False), **sampling)
    request = urllib.request.Request(args.base + "/v1/chat/completions",
                                     json.dumps(body).encode(), {"Content-Type": "application/json"})
    start = time.perf_counter()
    first = last = None
    usage = None
    content, reasoning = [], []
    finish = None
    with urllib.request.urlopen(request, timeout=1200) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[5:])
            if event.get("error"):
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if text or thought:
                    now = time.perf_counter()
                    first = now if first is None else first
                    last = now
                    content.append(text)
                    reasoning.append(thought)
                finish = choice.get("finish_reason") or finish
    end = time.perf_counter()
    if not usage or first is None or last <= first:
        raise RuntimeError("Missing usage or insufficient streaming tokens")
    after = metrics(args.base)
    diff = {k: after[k] - before.get(k, 0) for k in after}
    completed = total(diff, "request_success_total")
    if completed != 1:
        raise RuntimeError(f"Concurrent traffic detected: {completed} requests completed")
    drafts = total(diff, "spec_decode_num_drafts_total")
    accepted = total(diff, "spec_decode_num_accepted_tokens_total")
    n = usage["completion_tokens"]
    result = dict(label=args.label, task=task, seed=seed, sampling=sampling,
                  prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                  usage=usage, wall_s=end-start, ttft_s=first-start,
                  decode_tps=(n-1)/(last-first), end_to_end_tps=n/(end-start),
                  tokens_per_pass=1+accepted/drafts if drafts else None,
                  approx_step_ms=(last-first)/drafts*1000 if drafts else None,
                  spec_metric_delta=diff, finish_reason=finish,
                  content="".join(content), reasoning="".join(reasoning))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://localhost:8888")
    parser.add_argument("--model", default="GLM-5.3-EXL3")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--tasks", default=",".join(PROMPTS))
    parser.add_argument("--context-file")
    args = parser.parse_args()
    warm = run(args, "prose_beekeeper", 999, max_tokens=48)
    print(json.dumps(dict(warmup_decode_tps=warm["decode_tps"])), flush=True)
    rows = []
    with open(args.out, "a") as output:
        for repeat in range(args.repeats):
            for task in args.tasks.split(","):
                row = run(args, task, 1729 + repeat)
                rows.append(row)
                output.write(json.dumps(row) + "\n")
                output.flush()
                print(json.dumps({k: v for k, v in row.items() if k not in ("content", "reasoning", "spec_metric_delta")}), flush=True)
    for group in ("prose", "code"):
        values = [r["decode_tps"] for r in rows if r["task"].startswith(group)]
        if values:
            print(json.dumps(dict(group=group, count=len(values), mean=statistics.mean(values),
                                  median=statistics.median(values), min=min(values), max=max(values))), flush=True)


if __name__ == "__main__":
    main()
