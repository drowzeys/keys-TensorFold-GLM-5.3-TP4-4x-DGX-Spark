"""Daily-use bench: what a person or an agent actually does with the server, not single-shot prompts.

Five sessions, run as a client would (the whole conversation resent each turn, server defaults: thinking as the
template has it, default sampling, no draft-mode override), streamed so TTFT and decode speed are measured per turn:
  chat    6-turn conversation (questions, follow-ups, a rewrite)
  code    a ~6K-token Python module + 4 follow-up edit requests
  doc     a ~30K-token document + 3 questions about it
  agent   ~5K-token system prompt with tools + 4 tool-call/result steps
  quick   6 independent short questions (fresh conversations)
Per turn: prompt tokens, TTFT, completion tokens, decode tok/s, wall time. Totals: session wall time, median TTFT,
token-weighted decode tok/s. usage: daily.py URL LABEL OUT.json
"""
import json, statistics, sys, time, urllib.request

U, LABEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
MODE = sys.argv[4] if len(sys.argv) > 4 else ""      # optional "tf_mtp" for every request (e.g. auto); "" = server default
MAXTOK = 1024


def stream(messages, tools=None, seed=0):
    body = {"model": "glm-5.3-tf", "messages": messages, "max_tokens": MAXTOK, "stream": True, "seed": seed,
            "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
    if MODE:
        body["tf_mtp"] = MODE
    t0 = time.time(); first = last = None; text = reasoning = ""; calls = []; usage = {}
    req = urllib.request.Request(U + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "") + (d.get("reasoning_content") or "")
                if piece or d.get("tool_calls"):
                    now = time.time(); first = first or now; last = now
                text += d.get("content") or ""; reasoning += d.get("reasoning_content") or ""
                for tc in d.get("tool_calls") or []:
                    calls.append(tc)
    end = time.time()
    n = usage.get("completion_tokens", 0)
    dec = (n - 1) / (last - first) if first and last and last > first and n > 1 else 0.0
    return {"prompt": usage.get("prompt_tokens"), "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "ttft": (first or end) - t0, "completion": n, "decode_tps": dec, "wall": end - t0}, text, calls


def corpus(n_tokens, salt):
    import sysconfig
    from pathlib import Path
    root = Path(sysconfig.get_paths()["stdlib"])
    out, size = [], 0
    for p in sorted(root.glob("*.py")):
        t = p.read_text(errors="ignore"); out.append(f"# file {p.name} ({salt})\n{t}"); size += len(t)
        if size > n_tokens * 3.6:
            break
    return "".join(out)[: int(n_tokens * 3.6)]


def session(name, turns_fn):
    rows, t0 = [], time.time()
    turns_fn(rows)
    return {"session": name, "turns": rows, "wall": time.time() - t0}


def chat(rows):
    m = []
    for i, q in enumerate(["I'm planning a 4-day trip to Kyoto in late autumn. What should I prioritise?",
                           "I don't like crowds. Which of those can I do early morning, and what would you swap out?",
                           "Turn that into a day-by-day plan with rough times.",
                           "Day 3 looks tiring. Make it lighter and add one good place for dinner near Gion.",
                           "What should I pack for the weather?",
                           "Summarise the final plan in five bullet points."]):
        m.append({"role": "user", "content": q})
        st, text, _ = stream(m, seed=100 + i); rows.append(st)
        m.append({"role": "assistant", "content": text})


def code(rows):
    src = corpus(6000, "module")
    m = [{"role": "user", "content": "Here is a Python module I maintain:\n\n```python\n" + src +
          "\n```\n\nGive me a short review: the three most important problems."}]
    asks = ["Fix the first problem you found. Show only the changed functions.",
            "Now add type hints to those functions.",
            "Write pytest tests for the changed code.",
            "Explain in a few sentences what you changed and why, for the commit message."]
    st, text, _ = stream(m, seed=200); rows.append(st)
    for i, q in enumerate(asks):
        m += [{"role": "assistant", "content": text}, {"role": "user", "content": q}]
        st, text, _ = stream(m, seed=201 + i); rows.append(st)


def doc(rows):
    d = corpus(30000, "report")
    m = [{"role": "user", "content": "Read this source collection carefully:\n\n" + d +
          "\n\nWhich modules here deal with text processing? List them with one line each."}]
    st, text, _ = stream(m, seed=300); rows.append(st)
    for i, q in enumerate(["Which of those would you use to parse a CSV with quoted fields, and why?",
                           "Find two places where the code handles errors in an unusual way and explain them."]):
        m += [{"role": "assistant", "content": text}, {"role": "user", "content": q}]
        st, text, _ = stream(m, seed=301 + i); rows.append(st)


TOOLS = [{"type": "function", "function": {"name": n, "description": d, "parameters": {"type": "object",
          "properties": p, "required": list(p)}}} for n, d, p in [
    ("search_files", "Search the repository for a regex; returns matching lines with paths.", {"pattern": {"type": "string"}}),
    ("read_file", "Read a file from the repository.", {"path": {"type": "string"}}),
    ("run_tests", "Run the test suite or one test file; returns the summary.", {"target": {"type": "string"}}),
    ("edit_file", "Replace text in a file.", {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}})]]


def agent(rows):
    system = ("You are a coding agent working in a Python repository. Use the tools to investigate before answering. "
              "Repository guidelines follow.\n\n" + corpus(4500, "guidelines"))
    m = [{"role": "system", "content": system},
         {"role": "user", "content": "The test test_parse_dates fails with a timezone error. Find the cause and fix it."}]
    results = ["src/dates.py:41:    return datetime.strptime(s, FMT)\nsrc/dates.py:88:def parse_dates(rows):",
               "def parse_dates(rows):\n    out = []\n    for r in rows:\n        out.append(_parse(r['when']))\n    return out\n\n"
               "def _parse(s):\n    FMT = '%Y-%m-%dT%H:%M:%S'\n    return datetime.strptime(s, FMT)\n",
               "1 failed: test_parse_dates - ValueError: unconverted data remains: +02:00",
               "edited src/dates.py (1 replacement)\n\nAll 48 tests passed."]
    for i, res in enumerate(results):
        st, text, calls = stream(m, tools=TOOLS, seed=400 + i); rows.append(st)
        cid = f"call_{i}"
        name = (calls[0].get("function") or {}).get("name", "search_files") if calls else "search_files"
        m.append({"role": "assistant", "content": text, "tool_calls": [{"id": cid, "type": "function",
                  "function": {"name": name, "arguments": "{}"}}]})
        m.append({"role": "tool", "tool_call_id": cid, "content": res})
    st, _, _ = stream(m, tools=TOOLS, seed=499); rows.append(st)


def quick(rows):
    for i, q in enumerate(["What's the difference between a process and a thread?", "Convert 72 °F to Celsius.",
                           "Give me a regex for a UK postcode.", "Who wrote 'The Left Hand of Darkness'?",
                           "Write a haiku about debugging.", "How do I undo the last git commit but keep the changes?"]):
        st, _, _ = stream([{"role": "user", "content": q}], seed=500 + i); rows.append(st)


res = {"label": LABEL, "sessions": []}
for name, fn in (("chat", chat), ("code", code), ("doc", doc), ("agent", agent), ("quick", quick)):
    s = session(name, fn); res["sessions"].append(s)
    t = s["turns"]
    print(f"{LABEL} {name:6s} wall {s['wall']:7.1f} s  turns {len(t)}  TTFT median {statistics.median(x['ttft'] for x in t):6.2f} s "
          f"max {max(x['ttft'] for x in t):6.2f} s  decode {sum(x['completion'] for x in t) / max(1e-9, sum(x['completion'] / x['decode_tps'] for x in t if x['decode_tps'])):5.1f} tok/s",
          flush=True)
allt = [x for s in res["sessions"] for x in s["turns"]]
tot = {"wall": sum(s["wall"] for s in res["sessions"]), "ttft_median": statistics.median(x["ttft"] for x in allt),
       "decode_tps": sum(x["completion"] for x in allt) / sum(x["completion"] / x["decode_tps"] for x in allt if x["decode_tps"]),
       "tokens": sum(x["completion"] for x in allt)}
res["total"] = tot
print(f"{LABEL} TOTAL  wall {tot['wall']:.0f} s  TTFT median {tot['ttft_median']:.2f} s  decode {tot['decode_tps']:.1f} tok/s  "
      f"{tot['tokens']} tokens", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
