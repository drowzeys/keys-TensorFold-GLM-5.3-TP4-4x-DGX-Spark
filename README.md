# Full GLM-5.3 on four DGX Sparks with TensorFold (tensor parallel 4)

The full **GLM-5.3** (753B total, ~40B active) served natively by [TensorFold](https://github.com/ashhart/TensorFold)
across **four NVIDIA DGX Sparks** (GB10, one rank per Spark over ConnectX-7 RoCE). No vLLM in the serving path.
Image `ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-04-opt`. **The default needs no extra draft model**
(GLM-5.3's own MTP layer drafts); DFlash2 is an optional add-on.

**Abliterated weights:** [drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated)
(gated, automatic approval after the Responsible Use form). Stock parent:
[drowzeys/keys-GLM-5.3-EXL3-2.75BPW](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW). Method: [ablit/](ablit/).

## Performance (image `2026-10-04-opt`, thinking on)

Four DGX Sparks, the published image through `one-shot.sh`, default settings, **thinking on** (GLM-5.3 is a reasoning
model: we measure it the way it is meant to be used, and publish no thinking-off numbers).

| Summary | Speed |
|---|---|
| **Prose** (`./one-shot.sh bench`, greedy, 300 tokens, whole request) | **41.8 tok/s** |
| **Code** (same) | **38.1 tok/s** |
| **25K-token prompt** (prefill + answer, needle found) | **25.0 s** |
| **Prefill** at 128K tokens | **1,038 tok/s** (TTFT 125 s) |
| **4 streams, aggregate** (greedy, DFlash2): prose / code | **123.7 / 168.6 tok/s** (MTP: 118.5 / 134.0) |

| Single stream, thinking on | MTP = 2 (default, no extra model) | DFlash2 (optional draft) |
|---|---|---|
| Short context, greedy: prose / code | 37.7 / 41.4 tok/s | 36.1 / **47.7** tok/s |
| 32K context, sampled (T = 1.0): prose / code | 31.7 / 34.9 tok/s | **33.4 / 39.4** tok/s |
| Prefill 128K | 1,038 tok/s, TTFT 125 s | same |

The detailed rows use `bench/tfbench.py` (3 prose + 2 code prompts, 512 tokens; the 32K rows put ~32K tokens of
background text before each request).

| Concurrent streams, greedy (aggregate tok/s) | MTP = 2 | DFlash2 |
|---|---|---|
| **Prose, 1 stream** | 38.3 | **44.5** |
| **Prose, 4 streams** | **118.5** | **123.7** |
| **Code, 1 stream** | 44.2 | **67.2** |
| **Code, 4 streams** | **134.0** | **168.6** |
| Sampled (T = 1.0), 4 streams: prose / code | 69.3 / 71.0 | 75.5 / 63.6 |

`PARALLEL=4 CONTEXT=32768`, `tools/bench_concurrent.py --alone` (256-token replies; every concurrent reply checked equal
to the same request alone: 126 / 126; TTFT ≤ 0.93 s). Measured 2026-10-04 on this engine before its final memory
fixes; the prose prompts ran with thinking on, the code rows are raw code completions.

Memory: GPU memory on GB10 is unified. With the node settings (`vm.swappiness=1`) and `one-shot.sh` (page cache
dropped at start and while loading), each Spark keeps ~9-15 GB free while serving a 140K window with DFlash2 loaded,
and never swaps (traced every 5-10 s over short, 32K and 128K requests).

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
  over the four ranks; needles passed at 128K / 512K / ~1M on `2026-09-30`).
- Code: TensorFold fork branch [`drowzeys/TensorFold:glm53-tp4-opt`](https://github.com/drowzeys/TensorFold/tree/glm53-tp4-opt)
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
- [one-shot.sh](one-shot.sh) — `check`, `up`, `wait`, `bench`, `down`, `logs` for the prebuilt image.
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
