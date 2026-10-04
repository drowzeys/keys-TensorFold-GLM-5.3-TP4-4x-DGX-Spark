#!/usr/bin/env python3
"""Build the capture prompt corpus (prompts only — responses are generated ON-POLICY by the served target).
Sources (public Hugging Face datasets, downloaded by `datasets`; respect each dataset's terms):
  Aeala/ShareGPT_Vicuna_unfiltered (120,675 convs)  -> prose/chat/explanations (first human turn)
  mlabonne/open-perfectblend (1.42M rows, 'source')  -> code-ish sources (name match) + general instruct
  openai/gsm8k train (7,473) + EleutherAI/hendrycks_math (algebra train 1,744) -> reasoning (thinking on)
  templated list / essay / code tasks (taskbench.py + greedy_equiv.py style) -> the classes we benchmark
Mix (default N=6000): prose 40% | code 25% | list/structured 15% | reasoning 12% | essay 8%.
Output: prompts.jsonl {id, cls, think, max_tokens, text}. Token estimate uses the target tokenizer if available.
"""
import argparse, json, os, random, re
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=6000); ap.add_argument("--out", default="prompts.jsonl")
ap.add_argument("--seed", type=int, default=0); a = ap.parse_args(); rng = random.Random(a.seed)
from datasets import load_dataset
MIX = {"prose": 0.40, "code": 0.25, "list": 0.15, "reason": 0.12, "essay": 0.08}
MAXTOK = {"prose": 1024, "code": 1024, "list": 768, "reason": 1536, "essay": 1024}
def first_human(conv):
    for m in conv:
        if m.get("from") in ("human", "user") and m.get("value"): return m["value"].strip()
    return None
def ok(t): return t and 20 <= len(t) <= 1500 and not re.search(r"https?://|\bimage\b|\battached\b", t, re.I)
prose, code, general = [], [], []
sg = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered")["train"]
for i in rng.sample(range(len(sg)), min(40000, len(sg))):
    t = first_human(sg[i]["conversations"])
    if ok(t): prose.append(t)
pb = load_dataset("mlabonne/open-perfectblend")["train"]
srcs = {}
for i in rng.sample(range(len(pb)), 60000):
    r = pb[i]; s = r["source"]; srcs[s] = srcs.get(s, 0) + 1; t = first_human(r["conversations"])
    if not ok(t): continue
    (code if re.search(r"code|magicoder|evol|python|glaive|sql|leet", s, re.I) or re.search(r"\b(python|javascript|function|class|SQL|bash|regex|C\+\+|Rust|Go)\b", t) else general).append(t)
print("open-perfectblend sources sampled:", dict(sorted(srcs.items(), key=lambda x: -x[1])[:12]))
gsm = load_dataset("openai/gsm8k", "main")["train"]; mth = load_dataset("EleutherAI/hendrycks_math", "algebra")["train"]
reason = [r["question"] for r in gsm] + [r["problem"] for r in mth]
TOPICS = ["a lighthouse keeper in 1890s Norway", "a courier robot in a flooded Bangkok", "a violinist who loses her hearing", "a mountain village's last winter",
          "two brothers repairing a boat", "a night-shift nurse in Lagos", "a chess prodigy in exile", "a beekeeper during a drought", "a cartographer of a vanished coastline",
          "a translator at a border crossing", "a retired astronaut tending a garden", "a street musician in winter Montreal"]
LISTS = ["programming concepts", "world cities", "chemical compounds", "historical events", "musical instruments", "mathematical theorems", "plant species",
         "cooking techniques", "constellations", "programming languages", "board games", "economic terms", "rivers", "philosophers", "dog breeds", "cloud types"]
ESSAYS = ["Should cities ban private cars?", "Is remote work good for innovation?", "Should AI systems have a right to refuse tasks?", "Is nuclear power essential for decarbonization?",
          "Should voting be mandatory?", "Do social networks make us lonelier?", "Should space exploration be privatized?", "Is a four-day work week viable?",
          "Should schools teach personal finance before calculus?", "Is universal basic income a solution to automation?"]
CODES = ["an LRU cache with TTL expiry and thread safety", "a rate limiter (token bucket) with a decorator API", "a tiny JSON parser without using the json module",
         "a CSV to SQLite importer with schema inference", "a priority queue backed task scheduler", "a Markdown to HTML converter for headings, lists and code blocks",
         "a trie with prefix search and deletion", "an interval tree supporting overlap queries", "a minimal HTTP server that serves static files", "a Dijkstra shortest path on a weighted graph"]
def templ(cls, i):
    if cls == "list": return f"List {rng.choice([40, 60, 80, 120])} distinct {LISTS[i % len(LISTS)]}, numbered, each with a one-line description."
    if cls == "essay": return f"Write a structured argumentative essay (about 800 words, with an introduction, three sections with headers, and a conclusion) on: '{ESSAYS[i % len(ESSAYS)]}'"
    if cls == "code": return f"Write a complete, well-documented Python module implementing {CODES[i % len(CODES)]}, with a small CLI demo. Variant #{i}. Code only, no prose."
    if cls == "prose": return f"Write a vivid literary short story (at least 700 words) about {TOPICS[i % len(TOPICS)]}. No headings, flowing prose only."
out = []; counts = {}
for cls, frac in MIX.items():
    n = int(a.n * frac); pool = {"prose": prose, "code": code, "reason": reason, "list": [], "essay": []}[cls]
    n_templ = n if cls in ("list", "essay") else n // 4      # benchmark-class templates for prose/code too (25%)
    items = [templ(cls, i) for i in range(n_templ)] + rng.sample(pool, min(n - n_templ, len(pool)))
    if cls == "prose" and len(items) < n: items += rng.sample(general, min(n - len(items), len(general)))
    for t in items:
        think = cls == "reason" or rng.random() < 0.25        # 25% thinking-on elsewhere
        out.append({"id": len(out), "cls": cls, "think": think, "max_tokens": MAXTOK[cls] + (1024 if think else 0), "text": t})
    counts[cls] = len(items)
rng.shuffle(out)
with open(a.out, "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
print(f"wrote {len(out)} prompts -> {a.out}  {counts}  budget max_tokens sum = {sum(r['max_tokens'] for r in out):,}")
