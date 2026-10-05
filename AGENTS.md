# Instructions for coding agents: build and serve the fastest full GLM-5.3 on four DGX Sparks

You are setting up full GLM-5.3 (2.75 bpw EXL3) on **four** NVIDIA DGX Sparks (GB10) with TensorFold, tensor parallel
4. The prebuilt image already has the fastest measured settings as its defaults: **do not add tuning flags** unless a
step below says so. Work through the steps in order, check each step's result, and stop and report to your user
when a check fails. Never guess around a failure.

## Hard rules

- Four Sparks, one rank each, on a RoCE fabric (ConnectX-7 QSFP, 200G): a switch or direct cables. Fewer nodes do not
  fit the model.
- GB10 memory is unified: if a node runs out of memory it swaps until it hangs. **Never start the engine on a node
  that has other GPU jobs or less than ~100 GB available**. `./one-shot.sh check` reports both.
- Do not reboot a node to free memory; stop the other jobs and drop the page cache (`sync; sudo sysctl vm.drop_caches=3`).
- `vm.swappiness=1` on every node (node/gb10-node-settings.sh): with the default 60 the kernel swaps the engine out
  while it loads. Swap in use before a start: `sudo swapoff -a && sudo swapon -a`.
- `PARALLEL=N` holds N caches a rank: keep `CONTEXT=32768` with `PARALLEL=4`. Larger combinations can exhaust memory;
  the engine refuses ones it computes won't fit, but stay within this.
- Run commands from the orchestrating machine (any host with passwordless ssh to the four Sparks). It needs no GPU.

## Steps

1. **Inputs from the user**: the four Sparks' fabric addresses, rank 0 first (`NODES`); a checkpoint path that exists
   on all four (`MODEL`, e.g. an NFS export). Ask for these if you do not have them.
2. **Weights** (once, ~250 GB): `hf download drowzeys/keys-GLM-5.3-EXL3-2.75BPW --local-dir $MODEL` on the node that
   exports it, or on each node. Abliterated pack:
   `hf download drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated --local-dir $MODEL`
   ([card](https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-2.75BPW-Abliterated), gated auto-approve). On this fleet
   the live path is the ablit overlay `/mnt/spark2-models-local/GLM-5.3-EXL3-2.75-mixedK-EXL3NE-ablit`
   (`ablit/README.md`, `cluster.env`).
3. **Image** (once, on every node; if the pull cannot resolve `ghcr.io`, the node lost its DNS servers after a
   reboot: `sudo resolvectl dns <default-route interface> 1.1.1.1 8.8.8.8`): `docker pull ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-04-opt`.
4. **Node settings** (after every reboot, on every node): `sudo node/gb10-node-settings.sh`.
5. **Check**: `NODES=... MODEL=... ./one-shot.sh check` must end with `check: OK`. Fix each `FAIL`. Treat `WARN` lines
   as failures for a benchmark (other GPU jobs, low memory or compaction on all slow decode).
   - `check` finds both RoCE rails on each node (two PCIe twins of the QSFP port, by subnet). If it reports
     different subnets between ranks, the second rail is not addressed alike on all four: fix the addressing or use
     `RAILS=1` (about 2-4 % slower).
6. **Start**: `./one-shot.sh up`, then **immediately** `./one-shot.sh wait` (it also drops clean page cache on the nodes
   while they load: GB10's GPU allocations can only use free memory) (about 10 minutes: weights load, then decode graphs
   are captured). If a rank exits, `wait` prints its log; report it.
7. **Verify**: `./one-shot.sh bench` (thinking on). Expected on a healthy cluster (image `2026-10-04-opt`): prose about
   40 tok/s, code about 38 tok/s, the ~25K-token prompt answered in about 25 s, needle PASS. More than ~15 % below
   that means a node is the problem: rerun `check`, look for other processes, swap (`free -g`: swap must stay ~0) or
   heat (`nvidia-smi` on every node: clocks ~2.4 GHz, under ~85 °C).
8. **Report** the endpoint to the user: `http://<rank 0>:8890/v1`, model name `glm-5.3-tf`, OpenAI-compatible.
   It has no authentication: keep it on a private network.

## Options (only on the user's request)

- **DFlash2 drafts** (thinking on: +5 % prose / +13 % code at 32K, +15 % short code; needs ~1 GB more a rank): the user downloads
  [incoai/GLM-5.3-DFlash2](https://huggingface.co/incoai/GLM-5.3-DFlash2) themselves. It is licensed **CC BY-NC-ND
  4.0** (non-commercial, no derivatives); tell the user before using it. Then `DRAFT=<dir> ./one-shot.sh up`, and
  requests choose with `"tf_mtp": "dflash"` (or `"auto"`).
- **Their own fine-tuned draft**: `draft-finetune/README.md` (a further ~+10 % prose; same license as above).
- **Concurrent streams**: `PARALLEL=4 CONTEXT=32768 ./one-shot.sh up`.
- **1M-token context**: `CONTEXT=1000000` (decode context parallelism over the four ranks; slower prefill).
- **Reproducible long prompts**: `DOCKER_ENV="-e TF_EXL3_PROMPT_DET=slots16"` (~5 % slower prefill).

## Building from source instead of the image

Only if the user asks: `RECIPE.md`. The image is the tested configuration; a source build must reach the same
`bench` numbers before you report success.
