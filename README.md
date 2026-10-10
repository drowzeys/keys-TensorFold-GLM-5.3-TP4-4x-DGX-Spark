# Full GLM-5.3 on four DGX Sparks with TensorFold (tensor parallel 4)

The full **GLM-5.3** (753B total, ~40B active) served natively by [TensorFold](https://github.com/ashhart/TensorFold)
across **four NVIDIA DGX Sparks** (GB10, one rank per Spark over ConnectX-7 RoCE). No vLLM in the serving path.
Image `ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-09` (= `latest`). **The default needs no extra draft model**
(GLM-5.3's own MTP layer drafts). **New 2026-10-07:** an openly licensed DSpark drafter,
[drowzeys/keys-GLM-5.3-speculator.dspark-ft2](https://huggingface.co/drowzeys/keys-GLM-5.3-speculator.dspark-ft2)
(GLM-5.3 license), **+5 % prose / +12 % code** over MTP at 32K with thinking on ([below](#dspark-drafter-2026-10-07)).
DFlash2 remains an optional add-on.

> **Upgrade from `2026-10-04` / `2026-10-04-opt` / `2026-10-05` / `2026-10-07`:** `2026-10-09` fixes a hang that
> agent clients (title requests, cancelled requests) could trigger on all four ranks, and a stall that held the server
> for hours. `2026-10-04` also lacks the thinking-off fix and the stop-on-disconnect fix. Pull `2026-10-09` on every
> node and restart; nothing else changes ([what changed](#what-changed-in-2026-10-09)).

**Abliterated weights:** [drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated)
(gated, automatic approval after the Responsible Use form). Stock parent:
[drowzeys/keys-GLM-5.3-EXL3-2.75BPW](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW). Method: [ablit/](ablit/).

## Performance (image `2026-10-05`, thinking on)

Four DGX Sparks, the published image through `one-shot.sh`, default settings, **thinking on** (GLM-5.3 is a reasoning
model: we measure it the way it is meant to be used, and publish no thinking-off numbers).

| Summary | Speed |
|---|---|
| **Prose** (`./one-shot.sh bench`, greedy, 300 tokens, whole request) | **40.2 tok/s** |
| **Code** (same) | **39.0 tok/s** |
| **25K-token prompt** (prefill + answer, needle found) | **25.3 s** |
| **Prefill** at 128K tokens | **1,038 tok/s** (TTFT 125 s) |
| **Turn 2 of a 93K-token conversation** (prompt reuse) | **0.85 s** to first token (85.7 s without) |
| **4 streams, aggregate** (greedy chat, thinking on): prose / code | **78.7 / 100.2 tok/s** |
| **1M context**: needle at 905K tokens | **PASS**, decode **33.7 tok/s** at that depth |

| Single stream, thinking on | MTP = 2 (default, no extra model) | DFlash2 (optional draft) |
|---|---|---|
| Short context, greedy: prose / code | 37.7 / 41.4 tok/s | 36.1 / **47.7** tok/s |
| 32K context, sampled (T = 1.0): prose / code | 31.7 / 34.9 tok/s | **33.4 / 39.4** tok/s |
| Prefill 128K | 1,038 tok/s, TTFT 125 s | same |

The detailed rows use `bench/tfbench.py` (3 prose + 2 code prompts, 512 tokens; the 32K rows put ~32K tokens of
background text before each request).

### DSpark drafter (2026-10-07)

[drowzeys/keys-GLM-5.3-speculator.dspark-ft2](https://huggingface.co/drowzeys/keys-GLM-5.3-speculator.dspark-ft2) is
[RedHatAI/GLM-5.3-speculator.dspark](https://huggingface.co/RedHatAI/GLM-5.3-speculator.dspark) fine-tuned on 800
thinking-on captures of the Abliterated target (held-out acceptance 3.712 -> 3.770). It is under the GLM-5.3 license
(MIT-style; keep the notice), so unlike DFlash2 it can be redistributed. It drafts **alone** (`--mtp-drafts 0`: the MTP
layer is not loaded) with the **confidence** stop policy at 0.3 - the default cost policy, calibrated at context 0,
drafts too deep at long context.

```sh
hf download drowzeys/keys-GLM-5.3-speculator.dspark-ft2 --local-dir /models/keys-GLM-5.3-speculator.dspark-ft2   # every node
DSPARK=/models/keys-GLM-5.3-speculator.dspark-ft2 ./one-shot.sh up && ./one-shot.sh wait
```

Measured in one session (image + source of `2026-10-07`, Abliterated weights, thinking on, T = 1.0, `bench/tfbench.py`):

| tok/s | MTP = 2 (default) | **DSpark ft2** |
|---|---|---|
| prose short / 32K / ~120K | 36.8 / 32.2 / 29.2 | **38.2 / 33.8 / 31.8** |
| code short / 32K / ~120K | 39.9 / 34.8 / 31.5 | **45.8 / 39.0 / 37.4** |
| prefill 32K / 128K | 1,185 / 1,045 | **1,219 / 1,093** |
| needle 120K / 1M boot, needle in an 830K-token prompt | found / found | found / **found** (prefill 2,351 s) |

One stream only (`PARALLEL=1`); for concurrent streams keep MTP. The published image was validated with no source
mounted: 32.7 prose / 38.6 code at 32K, the same as the source-mounted run within noise. Evidence: `evidence/2026-10-07/`.

### Prompt reuse (on by default from `2026-10-05`)

A follow-up turn, an identical resend or a new conversation with the same system prompt resumes from the kept prompt
state instead of reading everything again (`TF_GLM53_PROMPT_REUSE=1`, the image default; `=0` turns it off). Measured
through `one-shot.sh` on `2026-10-05`:

| | Without reuse | With reuse |
|---|---|---|
| Turn 2 of a 93K-token conversation (single stream) | 85.7 s | **0.85 s** |
| Identical resend of that 93K prompt | 85.7 s | **0.15 s** |
| New conversation, same 14.7K-token system prompt | 13.5 s | **0.38 s** |
| Same, 4 streams x 32K: turn 2 at 25K / resend / shared system prompt | 22.0 / 22.0 / 9.2 s | **0.69 / 0.04 / 0.28 s** |

A cold prompt costs the same as before (`one-shot.sh bench` 25.3 s on the 25K prompt against 25.0 s without reuse).
With the default (fastest) MoE prompt kernel a resumed prompt is a valid prefill, not byte-identical to a cold one;
`TF_EXL3_PROMPT_DET=slots16` makes them identical.

### Concurrent streams

| `PARALLEL=4 CONTEXT=32768`, greedy chat, thinking on (MTP drafts) | 1 stream | 4 streams (aggregate) |
|---|---|---|
| Prose | 34.7 tok/s | **78.7 tok/s** |
| Code | 39.1 tok/s | **100.2 tok/s** |

`bench/conc_chat.py` (the tfbench prompts as chat requests, 512 tokens; TTFT ≤ 0.64 s). Earlier we published 118.5 /
134.0 here: those ran prose with thinking off and code as raw completions, which is not how the model is used.

### 1M context (`CONTEXT=1000000`)

Decode context parallelism over the four ranks (one stream). Validated on `2026-10-04-opt` (same engine), thinking on,
a passphrase at half depth (`bench/needle.py`):

| Needle | Prompt | Time to answer | Prefill | Decode at depth |
|---|---|---|---|---|
| ~125K | 114,944 tokens | 4.9 min | 394 tok/s | **37.0 tok/s** |
| ~500K | 456,383 tokens | 21.6 min | 352 tok/s | **35.7 tok/s** |
| ~1M | 905,182 tokens | 49 min | 307 tok/s | **33.7 tok/s** |

All three answered exactly. Against the `2026-09-30` image: prefill +35 % at ~1M (228 → 307 tok/s), decode ~26 → 34-37
tok/s. Each Spark kept 15-16 GB free with no swap for the whole run.

### Same prompts as bertholomus' full GLM-5.3 TP4 recipe

[bertholomus/glm-5.3-tensorfold-tp4-4xgb10](https://github.com/bertholomus/glm-5.3-tensorfold-tp4-4xgb10) (TensorFold,
EXL3 3.0 bpw, MTP-3, no DFlash2) publishes its four `bench/tf_greedy.py` prompts: greedy, 300 tokens, timed over the
whole request (they ran thinking off). Ours, same prompts and timing, **thinking on** (`bench/h2h.py`, 2026-10-04):

| Prompt | Ours, MTP = 2 | Ours, DFlash2 | bertholomus (their published numbers, thinking off, their hardware) |
|---|---|---|---|
| Hash table explanation | 41.1 | **48.6** | 34.6 (release) · 39.6 (best) |
| ISO 8601 parser + tests | 40.8 | **52.0** | 40.6 · 48.1 |
| Planets list | 43.1 | **50.4** | 42.1 · 48.4 |
| French translation | 42.6 | **53.6** | 38.1 · 43.6 |

Their numbers are from their repository, not re-measured here. Their quant is larger (3.0 bpw, KL 0.109 vs our 0.124:
better quality), so this compares recipes, not equal-quality builds.

## What changed in `2026-10-09`

Fixes only (same engine, kernels and defaults as `2026-10-07`; source: `2026-10-07` + three commits).

- **Fix: all four ranks hung for good under agent traffic** (field report on `2026-10-05`, ~1 h 45 of Hermes-style
  traffic; also the likely cause of a third-party Spark-Bench run whose last scenarios timed out). A background
  request (e.g. a conversation-title request) cut by a foreground one, whose client left while it waited to resume,
  handed the engine's turn back twice; a second request then entered the engine beside the first, the ranks ran
  mismatched collectives, and the HTTP server stopped accepting. Now the turn is only given back by the request that
  holds it, the four-rank engine never yields a running reply, and it never takes two requests at once.
- **Sealed one-stream messages**: every message rank 0 sends the others carries a sequence number and checksum; a rank
  out of step stops with a named error instead of running another request's message.
- **Stall watchdog on by default** (`TF_GLM_MULTI_WATCHDOG_S`, default **900** s, `0` = off; was off): a decode round
  or prompt chunk that stalls that long dumps every thread's stack to the rank's log and exits the rank; the other
  ranks follow. It is re-armed every round and every prompt chunk, and disarmed while the server is idle. Keep it above
  your longest prompt chunk (~8 s) and round; the default leaves a wide margin.
- **Stream write timeout** (`TF_STREAM_WRITE_TIMEOUT_S`, default **120** s, `0` = off): a streaming client that stops
  reading without closing its connection is treated as gone, instead of holding the engine.
- **`./one-shot.sh watch`**: because ranks now exit on a stall instead of hanging, run `watch` beside an unattended
  server. When any rank exits it saves the four logs to `logs-<time>/`, takes the server down and starts all four again
  (`up` + `wait`); after `MAX_RESTARTS` (3) restarts within an hour it gives up and leaves the server down. Ranks are
  never restarted one at a time: a single rank cannot rejoin the others.

Set either variable with `DOCKER_ENV`, e.g. `DOCKER_ENV="-e TF_GLM_MULTI_WATCHDOG_S=1800" ./one-shot.sh up`.

Verified on four Sparks at `CONTEXT=163840`: the field repro (title request cut, its client leaving while it waits,
a second request arriving) completes both foreground replies; streamed and non-streamed client disconnects free the
engine within one round; thinking-off prompts end in `<think></think>`; a rank frozen with `docker pause` makes the
others exit after the watchdog (60 s in the test). Evidence: `evidence/2026-10-09/`.

## What changed in `2026-10-05`

- **Prompt reuse on by default**: kept prompt states (system prompt, earlier turns, identical resends) at assistant
  openers, at most one copy a conversation, nothing extra on a cold prompt; a resend after other requests replays.
  Built after MiaAI-Lab's 0008 / 0015 / 0042 / 0063.
- **Copy drafts** (prompt-lookup drafts, after MiaAI-Lab 0007 / 0032) are in the engine but **off by default**
  (`TF_GLM53_COPY_DRAFTS=1` to try): not yet validated on a live server.
- **Memory hardening** (below): `one-shot.sh up` waits for every node to give back the memory of a lane stopped seconds
  earlier, and `wait` reports each node's headroom.

## What changed in `2026-10-04-opt`

Against `2026-10-04` (same day, back to back): an MTP pass 76 → 61 ms, 32K prose 25.8 → 31-33 tok/s, 32K code
32.8 → 39 tok/s; 128K prefill 1,000 → 1,038 tok/s. Plus fixes found while validating the published path: thinking-off
requests were still reasoning (fixed), and the node could swap the engine out while it loaded (fixed: below).
Every change below is on by default, keeps replies bit-exact (drafted == serial, concurrent == alone: checked by the
test suite), and has an environment switch to turn it off.

| Change | From |
|---|---|
| Routed-expert decode kernel: 16-byte non-coherent trellis loads, fused down + combine (−5-6 % a layer) | MiaAI-Lab 0047 / 0016 (0047 after Jay Leaton) |
| L2 prefetch of the next kernels' weights during each layer's all-reduce | MiaAI-Lab 0046 (after Jay Leaton) |
| Second CUDA stream in decode: key path beside the query path, shared expert beside the routed experts | bertholomus |
| MTP draft steps reuse the first step's DSA token selection | bertholomus |
| 4-bit copy of the output head for draft steps (verification keeps BF16) | TensorFold `glm5_next`, MiaAI-Lab |
| Finer index-key graph buckets (a 32K context scores 40,960 keys instead of 65,536) | MiaAI-Lab 0043 idea |
| Both PCIe twins of the QSFP port as RoCE rails, matched by subnet | MiaAI-Lab |
| Sampled verify exchanges each rank's top candidates, not full logits; small gathers over RoCE | TensorFold `glm5_next`, MiaAI-Lab 0034 |
| Rank 0's stop (client gone, stop string) ends every rank in the same round at `--parallel 1` (bug fix) | MiaAI-Lab 0070 |
| Tokens stream one round late; DFlash2 candidates in one pinned copy (less rank-0 skew) | MiaAI-Lab 0013 |
| 4 streams: draft cut by chain probability, every graph captured at warm-up, short prompts fill before the round | bertholomus |
| Sealed rank messages, `/health` `iteration_s`, optional stall watchdog | MiaAI-Lab 0065 |
| **Fix:** loading on GB10: each shard leaves the page cache once read, `vm.swappiness=1`, `one-shot.sh` drops page cache at start and during load, finer index buckets off by default (their graphs cost ~10 GB a rank) | ours |
| **Fix:** `enable_thinking: false` now really turns thinking off (GLM-5.3's template always opens `<think>`; the CUDA server did not close it, so earlier images reasoned anyway) | ours |

Tried and **not** default: finer index buckets (`TF_GLM53_INDEX_SPLIT=4`: no faster, ~10 GB of graphs a rank), MTP depth 3 (slower: 29.0 vs 31.0 at 32K), 4 NCCL channels (32K prefill 1,090 → 733),
FP8 verify head (lossy; `TF_GLM53_VERIFY_HEAD=fp8`), prompt-state reuse (`TF_GLM53_PROMPT_REUSE=1`: turn 2 of a 60K
conversation 57.7 s → 0.74 s TTFT, a shared 14.7K system prompt 14.4 s → 0.49 s; off until its cold-prompt cost is
measured cleanly and a resend fix is validated).

## Quick start (prebuilt image)

```sh
# every Spark, after each boot
sudo node/gb10-node-settings.sh
# from any machine that can ssh to the four Sparks
export NODES="spark1 spark2 spark3 spark4" MODEL=/models/GLM-5.3-EXL3-2.75BPW
# abliterated: hf download drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated --local-dir $MODEL
# KeySpark live overlay: source cluster.env
./one-shot.sh check && ./one-shot.sh up && ./one-shot.sh wait && ./one-shot.sh bench
curl http://spark1:8890/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "glm-5.3-tf", "messages": [{"role": "user", "content": "Hello"}]}'
```

- **Coding agents** (Claude Code, Codex, ...): point yours at [AGENTS.md](AGENTS.md); it builds and verifies the fastest
  configuration step by step.
- DFlash2 (optional): download [incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) yourself (CC
  BY-NC-ND 4.0: non-commercial, no derivatives), `DRAFT=/models/GLM-5.3-DFlash2 ./one-shot.sh up`, and send
  `"tf_mtp": "dflash"` (or `"auto"`) per request.
- 4 streams: `PARALLEL=4 CONTEXT=32768 ./one-shot.sh up`. 1M context: `CONTEXT=1000000` (decode context parallelism
  over the four ranks, one stream; needles pass at 115K / 456K / 905K).
- **Agents (Hermes, OpenClaw, ...)**: `PARALLEL=4 CONTEXT=32768` with prompt reuse (default). An agent resends a large
  system prompt and history every step; reuse makes each step start in ~0.15-1 s instead of ~16 s, and 4 streams stop
  side requests from queueing. Set the agent's context to 64K or less (Hermes: `context_length: 64000`, `max_tokens:
  8192`, `compression.threshold: 0.4` keeps a conversation under 32K). For long documents use the 1M lane instead.
- Code (image `2026-10-09`): TensorFold fork branch [`drowzeys/TensorFold:glm53-tp4-2026-10-09`](https://github.com/drowzeys/TensorFold/tree/glm53-tp4-2026-10-09) (the hang fixes on top of `2026-10-07`)
- Code (image `2026-10-07`): TensorFold fork branch [`drowzeys/TensorFold:glm53-tp4-2026-10-07`](https://github.com/drowzeys/TensorFold/tree/glm53-tp4-2026-10-07) (DSpark + copy drafts on top of `2026-10-05`)
- Code: TensorFold fork branch [`drowzeys/TensorFold:glm53-tp4-2026-10-05`](https://github.com/drowzeys/TensorFold/tree/glm53-tp4-2026-10-05)
  ([PR #159](https://github.com/ashhart/TensorFold/pull/159) carries the base engine). Weights:
  [drowzeys/keys-GLM-5.3-EXL3-2.75BPW](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW) (KL 0.124 nats /
  top-1 89.6 % vs BF16). Abliterated pack:
  [drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated)
  (Blackfrost derisk from [keys-GLM-5.3-EXL3-Abliterated](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-Abliterated)
  re-quantized to K5; gate **31/32** refusal, **22/22** cyber, thinking off; [ablit/](ablit/)). From source: [RECIPE.md](RECIPE.md).

## Reproducibility

Decode is row-invariant (rank-order sums, no atomics): a drafted reply equals a serial one and concurrent replies equal
the same request alone. Prompts longer than one chunk (~4K tokens) use the MoE prompt kernel, which by default adds a
row's expert outputs with fp32 atomics in arrival order (fastest prefill; the same long prompt can give a different
reply run to run). `DOCKER_ENV="-e TF_EXL3_PROMPT_DET=slots16"` makes long prompts reproducible (~4-6 % slower prefill).

## Build your own draft (optional, +10 % prose with DFlash2)

Fine-tuning the DFlash2 draft on this checkpoint's own outputs raised 32K prose 27.7 → 30.6 and code 34.3 → 37.0 tok/s
on our `2026-10-03` build. We cannot publish that draft (incoai's CC BY-NC-ND-4.0 license); [draft-finetune/](draft-finetune/)
builds your own for non-commercial use.

## Memory safety on GB10 (built into the recipe)

GB10's GPU memory is the system's unified memory: a node that runs short swaps the engine out, and a node that runs
out stalls until the watchdog reboots it. What the recipe does about it, and why:

| Step | Where | Why |
|---|---|---|
| `vm.swappiness=1` | `node/gb10-node-settings.sh` (every boot); `check` fails without it | with the default 60 the kernel swapped the engine out while it loaded, rather than drop the checkpoint's page cache (swap full with 25+ GB still "available") |
| Each shard leaves the page cache once read | the engine (`posix_fadvise` after every layer) | ~250 GB of checkpoint streams through the page cache, which GB10's GPU allocations cannot use |
| Page cache dropped at `up`, and every 15 s during `wait` | `one-shot.sh` (passwordless sudo; `check` warns without it) | the node exporting the checkpoint over NFS caches what it serves the other ranks; an image pull leaves tens of GB cached |
| `up` waits until every node is back near full memory | `one-shot.sh` | a lane started seconds after another stopped came up with ~1 GB free after warm-up (2026-10-05) |
| `wait` prints each node's free memory, warns under 6 GB | `one-shot.sh` | catch a bad start before it serves |
| `check` warns on swap in use, other GPU processes, < 100 GB free | `one-shot.sh` | anything else on a rank also slows every all-reduce |
| Decode graphs: index-key buckets in powers of two | engine default (`TF_GLM53_INDEX_SPLIT=1`) | finer buckets cost ~10 GB of graphs a rank for no measurable speed |

Measured with these: ~9-15 GB free on every Spark while serving (140K + DFlash2, 4 x 32K with prompt reuse, or 1M),
never any swap.

## Node notes that cost us time

- `vm.compaction_proactiveness=0` on every node (compaction stalls a rank ~130 ms at a time).
- Anything else on a rank (a browser, other containers, a memory-starved page cache) shows up as rank skew: all four
  wait for the slowest at every all-reduce. `./one-shot.sh check` flags it. Our own numbers dropped ~15 % on a
  memory-pressured afternoon and came back after `drop_caches` + compaction.
- RoCE GID indices and PCIe enumeration can move across reboots: `one-shot.sh` looks both up at every start.
- `--parallel 4` needs `--context 32768` (each stream holds its own cache); the engine refuses combinations that would
  not fit (GB10's unified memory swaps instead of failing).

## This repo

- [AGENTS.md](AGENTS.md) — step-by-step for coding agents (and humans).
- [one-shot.sh](one-shot.sh) — `check`, `up`, `wait`, `bench`, `watch`, `down`, `logs` for the prebuilt image.
- [ablit/](ablit/) — Blackfrost derisk overlay of the 2.75 bpw checkpoint
  ([keys-GLM-5.3-EXL3-2.75BPW-Abliterated](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated));
  [cluster.env](cluster.env) — this fleet's `NODES`/`MODEL`.
- [RECIPE.md](RECIPE.md) — from source; [node/](node/) — node settings; [draft-finetune/](draft-finetune/) — own draft.
- [bench/](bench/) — every bench behind these tables; [evidence/](evidence/) — logs and raw results by date.

## Credits

This recipe stands on other people's work. Thank you to:

- **[Z.ai](https://huggingface.co/zai-org)** — GLM-5.3, its weights and its MTP layer (weights under the GLM-5.3 license).
- **[Ash Hart / TensorFold](https://github.com/ashhart/TensorFold)** — the engine (Apache-2.0 from 0.6.0; earlier code
  MIT): the CUDA server, the EXL3 decode and expert kernels, the `glm5_next` family this engine builds on, exact
  drafted decoding. And the TensorFold contributors, among them nood-co1, vcruz305, MiaAI-Lab, philip-pentatonic,
  tournierjc, mikolaj92, jschmied, shantanugoel, plotarmordev, cshintov, akol1, chadhurley25075-png, di37, gilby,
  gprot42, EugeneClaw, Chedrian07, eleqtrizit, jayleaton, kky42, feni6, taussoe, MovieMaker93 and mgoldwasser.
- **[MiaAI-Lab / Mia's AI Lab](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)** (Apache-2.0)
  — the GLM prompt-kernel designs our prompt experts adapt, and the patches behind most of this image's levers: 16-byte
  trellis loads (0047), fused expert epilogues (0016), L2 prefetch (0046, its kernel used unchanged), the FP8 head
  (0002), the late token stream and pinned candidates (0013), the nucleus union (0034), the stop vote (0070), rank
  checks (0065), the dual-rail setup, the visible-key idea (0043) and the prompt-reuse designs (0008 / 0015 / 0042 /
  0063). Their [full GLM-5.3 3-Spark recipe](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold) is
  where we found the DSpark drafter and the NVMe prompt cache.
- **[Jay Leaton / glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark)** (Apache-2.0) — the
  L2 prefetch (patch 0460) and 16-byte trellis loads (patch 0580) that MiaAI-Lab's 0046 / 0047 adapt.
- **[BertholomusAI (Albert Lee)](https://github.com/bertholomus/glm-5.3-tensorfold-tp4-4xgb10)** (Apache-2.0) — the
  decode side stream, MTP index reuse, the draft-depth policy, the draft cut and warm-up graph capture for concurrent
  streams, quick fills (ideas re-implemented here), and the four head-to-head prompts.
- **[turboderp / ExLlamaV3](https://github.com/turboderp-org/exllamav3)** (MIT) — the EXL3 format and converter.
- **vcruz305** — the per-expert mixed-width EXL3 work this checkpoint's quantization builds on.
- **cuda-exl3** — the stacked expert layout and grouped GEMM behind the prompt path.
- **[b12x / local-inference-lab](https://github.com/local-inference-lab/b12x)** (Apache-2.0) — the RoCE one-shot
  all-reduce and all-gather.
- **[incoai](https://huggingface.co/incoai)** — the optional DFlash2 draft (CC BY-NC-ND 4.0; never redistributed here).
- **NVIDIA** — DGX Spark, the PyTorch container, CUDA and NCCL.
- **[Anthropic's Claude](https://claude.com/claude-code)** (Claude Code) — engineering assistance: ports, tests,
  benchmarks and documentation, as the commits' co-author lines record.

Full notices: [NOTICE.md](NOTICE.md) and TensorFold's `THIRD_PARTY_NOTICES.md`. Scripts here: MIT (see LICENSE).
Model weights: the GLM-5.3 license of the base model.
