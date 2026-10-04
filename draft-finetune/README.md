# Build your own draft (optional): an on-policy DFlash2 fine-tune

The public [incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) draft was trained against BF16
GLM-5.3. Fine-tuned on the 2.75 bpw target's own outputs, it predicts that target better and speeds prose up by
about 10 % on our build. **We cannot share our fine-tuned draft:** it is a derivative of incoai's draft, which is
licensed CC-BY-NC-ND-4.0 (no derivatives may be distributed; non-commercial). These scripts let you build your own
for your own non-commercial use. Check the license terms for your situation.

## What it gave us

Four DGX Sparks, the same serving code (image `2026-10-03`), default settings, 3-repeat means:

| | Stock incoai draft | Our fine-tune | Gain |
|---|---|---|---|
| **32K context, prose** (T = 1.0) | 27.7 tok/s | **30.6 tok/s** | **+10.5 %** |
| **32K context, code** (T = 1.0) | 34.3 | **37.0** | **+7.8 %** |
| Short context, prose (greedy) | 31.0 | **32.8** | +6 % |
| Short context, code (greedy) | **40.0** | 38.6 | -3 % |
| 4 streams, chat / code (greedy, aggregate) | 69.7 / 95.4 | **75.0 / 96.4** | +8 % / +1 % |

Measured one day apart on the same cluster. Prose gains most: the stock draft's chains break where the 2.75 bpw
target's choices differ from BF16's (~9 % of tokens). Our fine-tune used 900 captured sequences; a second one on
3.2x the data (2,900 sequences) gave no further gain, so the draft's capacity, not data, is the limit after that.

## Steps

You need the four Sparks serving (for capture) and then one Spark free for ~4 h (for training).

**1. Prompts.** Builds a 6,000-prompt mix (prose 40 %, code 25 %, list 15 %, reasoning 12 %, essay 8 %) from public
Hugging Face datasets (ShareGPT_Vicuna_unfiltered, open-perfectblend, gsm8k, hendrycks_math) plus templated tasks.
Responses are not taken from the datasets: the served target writes them in step 2.

```sh
pip install datasets
python3 build_prompts.py --n 6000 --out prompts.jsonl
```

**2. Capture on-policy data** from your running server. Rank 0 writes one `seq-*.safetensors` per request: the
target's tap rows (fp8) and its top-32 soft labels for every token it sampled.

```sh
mkdir -p /data/cap   # on the rank-0 Spark (~29 GB for 900 requests)
NODES="spark1 spark2 spark3 spark4" MODEL=/models/GLM-5.3-EXL3-2.75BPW DRAFT=/models/GLM-5.3-DFlash2 \
  DOCKER_ENV="-e TF_GLM53_CAPTURE_DIR=/cap -v /data/cap:/cap" ../one-shot.sh up
python3 capture_corpus.py --url http://spark1:8890 --prompts prompts.jsonl     # 900 prompts, ~9 h, resumable
../one-shot.sh down
```

Capture needs a draft loaded (`DRAFT=`): that is what makes the target record its tap layers.

**3. Train** on one Spark that is not serving (peak ~55 GB; ~4 h for 3,200 steps at batch 2):

```sh
DRAFT=/models/GLM-5.3-DFlash2 TARGET=/models/GLM-5.3-EXL3-2.75BPW CAP=/data/cap OUT=/data/draft-ft ./train.sh
```

The trainer prints `[data] train seqs=N ... from /seq` early on (check N), evaluation every 100 steps, and a
`[holdout-final]` line with per-slot top-1 and acceptance length at the end. Output: `/data/draft-ft/model.safetensors`
+ `config.json`, the same layout as incoai's draft.

**4. Serve with it:**

```sh
NODES="spark1 spark2 spark3 spark4" MODEL=/models/GLM-5.3-EXL3-2.75BPW DRAFT=/data/draft-ft ../one-shot.sh up
```

Copy `/data/draft-ft` to the same path on all four Sparks first (or put it on the shared model export).

## Files

| File | What |
|---|---|
| `build_prompts.py` | the prompt mix (step 1) |
| `capture_corpus.py` | drives the capture against the running server (step 2) |
| `dflash2_train.py` | the trainer: KL to the target's soft labels on every draft slot + the DFlash2 candidate-selector loss (step 3) |
| `train.sh` | runs the trainer in the prebuilt image with the settings of our fine-tune (step 3) |

Notes: capture runs on the single-stream path (not with `--parallel`). Retrain after any change to the target
weights: the draft learns that exact target. Scripts here are MIT; the draft you produce stays under incoai's license.
