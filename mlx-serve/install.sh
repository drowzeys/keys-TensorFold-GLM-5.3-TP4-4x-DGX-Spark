#!/bin/bash
# From nothing to ready-to-serve: full GLM-5.3 (EXL3 2.75 bpw) on four DGX Sparks, from public sources only.
#   mlx-serve/install.sh <spark1> <spark2> <spark3> <spark4>     # the four fabric (RoCE) addresses, rank 0 first
#   mlx-serve/install.sh                                         # again later: reads mlx-serve/my-cluster.env, resumes
# then:  mlx-serve/one-shot.sh up && mlx-serve/one-shot.sh bench
# Run it from any machine that reaches the four Sparks with key-based ssh (one of the Sparks is fine). It
#   1. checks each Spark: ssh, GB10 and its memory, Docker with the NVIDIA runtime, the RoCE devices, disk space
#   2. applies the two kernel settings a rank needs (node/gb10-node-settings.sh; they last until a reboot)
#   3. pulls the image on each Spark and checks its digest
#   4. downloads the checkpoint (277 GB, 91 files) and the 6-bit output head to MODEL on each Spark; a broken
#      download resumes, files already there are kept
#   5. writes mlx-serve/my-cluster.env (NODES, MODEL, IMAGE) and runs `one-shot.sh check`
# The drafter (DSpark ft2) is inside the image; nothing else is downloaded.
# Settings (environment):
#   MODEL    [~/models/GLM-5.3-EXL3-2.75BPW] the checkpoint directory, the same path on all four
#   SHARED=1 MODEL is one shared directory (NFS) seen by all four: download once, from rank 0
#            (SHARED_HOST=<address> when another machine is the one that can write to it)
#   WEIGHTS  [drowzeys/keys-GLM-5.3-EXL3-2.75BPW] the Hugging Face repository      SKIP_WEIGHTS=1 leaves step 4 out
#   VERIFY=1 also checks every weight file's sha256 against the repository (about 10 minutes a Spark)
#   IMAGE, DIGEST   the image and the digest it must have (DIGEST= accepts any)
set -u
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(dirname "$HERE")
[ $# = 0 ] && [ -f $HERE/my-cluster.env ] && . $HERE/my-cluster.env
[ $# = 0 ] || NODES="$*"
read -r -a N <<<"${NODES:-}"
[ ${#N[@]} = 4 ] || { sed -n 2,9p "$0"; exit 2; }
IMAGE=${IMAGE:-ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-12-r2-mlx-serve}
DIGEST=${DIGEST-sha256:6290c15f1596347468fa63c35f9e5ebc37212294fd283fc1c7165fddbcb9b34d}
WEIGHTS=${WEIGHTS:-drowzeys/keys-GLM-5.3-EXL3-2.75BPW}; MODEL=${MODEL:-}; NEED_GB=${NEED_GB:-280}
rsh() { ssh -o ConnectTimeout=10 -o BatchMode=yes -o ServerAliveInterval=30 "$@"; }
say() { printf '==> %s\n' "$*"; }; die() { printf '  FAIL %s\n' "$*" >&2; exit 1; }
LOG=$HERE/run-install; mkdir -p $LOG

say "1/5 the four Sparks"
bad=0
for r in 0 1 2 3; do n=${N[$r]}
  out=$(rsh $n '
    echo home=$HOME
    echo gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr " " _)
    echo arch=$(uname -m) mem_gb=$(awk "/MemTotal/{print int(\$2/1048576)}" /proc/meminfo)
    docker info >/dev/null 2>&1 && echo docker=ok || echo docker=no
    { docker info 2>/dev/null | grep -qi "nvidia" || command -v nvidia-ctk >/dev/null || command -v nvidia-container-cli >/dev/null; } && echo gpu_in_docker=ok || echo gpu_in_docker=no
    [ -d /dev/infiniband ] && command -v ibv_devices >/dev/null && echo rdma=ok || echo rdma=no
    sudo -n true 2>/dev/null && echo sudo=ok || echo sudo=no
    command -v python3 >/dev/null && command -v curl >/dev/null && echo tools=ok || echo tools=no' 2>&1) \
    || { echo "  FAIL rank $r ($n): no key-based ssh (ssh-copy-id $n, then: ssh -o BatchMode=yes $n true)"; bad=1; continue; }
  echo "  rank $r ($n): $(tr '\n' ' ' <<<"$out")"
  [ -n "$MODEL" ] || MODEL="$(grep -o 'home=.*' <<<"$out" | cut -d= -f2)/models/GLM-5.3-EXL3-2.75BPW"
  grep -q "gpu=.*GB10" <<<"$out" || echo "  WARN $n is not a GB10: only DGX Spark has been run"
  grep -q "arch=aarch64" <<<"$out" || { echo "  FAIL $n is not aarch64: the image is built for DGX Spark"; bad=1; }
  m=$(grep -o 'mem_gb=[0-9]*' <<<"$out" | cut -d= -f2); [ "${m:-0}" -ge 110 ] || { echo "  FAIL $n has ${m} GB of memory: a rank needs a 128 GB Spark"; bad=1; }
  grep -q docker=ok <<<"$out" || { echo "  FAIL $n: docker does not run for this user (sudo usermod -aG docker \$USER, log in again)"; bad=1; }
  grep -q gpu_in_docker=ok <<<"$out" || { echo "  FAIL $n: Docker cannot give a container the GPU (install nvidia-container-toolkit)"; bad=1; }
  grep -q rdma=ok <<<"$out" || { echo "  FAIL $n: no /dev/infiniband or ibv_devices (sudo apt install rdma-core ibverbs-utils; the ranks talk over RoCE)"; bad=1; }
  grep -q tools=ok <<<"$out" || { echo "  FAIL $n: python3 and curl are needed for the weight download"; bad=1; }
  grep -q sudo=ok <<<"$out" || echo "  WARN $n: no passwordless sudo: run 'sudo node/gb10-node-settings.sh' there yourself after each boot, and 'sudo sysctl vm.drop_caches=1' while the ranks load"
done
[ $bad = 0 ] || die "fix the lines above and run install.sh again"

say "2/5 kernel settings (vm.compaction_proactiveness=0, vm.swappiness=1; until the next reboot)"
for n in "${N[@]}"; do rsh $n 'sudo -n sysctl -q -w vm.compaction_proactiveness=0 vm.swappiness=1 2>/dev/null && echo "  '$n': set" || echo "  '$n': not set (no passwordless sudo)"'; done

say "3/5 image $IMAGE"
for n in "${N[@]}"; do rsh $n "docker pull -q $IMAGE" > $LOG/pull-$n.log 2>&1 & done; wait
for n in "${N[@]}"; do d=$(rsh $n "docker image inspect $IMAGE --format '{{index .RepoDigests 0}}'" 2>/dev/null)
  [ -n "$d" ] || die "$n did not pull the image: $(tail -1 $LOG/pull-$n.log)"
  [ -z "$DIGEST" ] || [ "${d##*@}" = "$DIGEST" ] || die "$n has ${d##*@}, expected $DIGEST (DIGEST= accepts any)"
  echo "  $n: ${d##*@}"; done

if [ "${SKIP_WEIGHTS:-0}" = 1 ]; then say "4/5 weights: skipped (MODEL=$MODEL)"; else
  say "4/5 weights $WEIGHTS -> $MODEL$([ "${SHARED:-0}" = 1 ] && echo " (shared: ${SHARED_HOST:-rank 0} downloads)")"
  targets=("${N[@]}"); [ "${SHARED:-0}" != 1 ] || targets=("${SHARED_HOST:-${N[0]}}")
  for n in "${targets[@]}"; do
    f=$(rsh $n "mkdir -p '$MODEL' && df --output=avail -BG '$MODEL' | tail -1 | tr -dc 0-9; echo; du -sBG '$MODEL' | cut -f1 | tr -dc 0-9" | paste -sd' ')
    [ $(( ${f% *} + ${f#* } )) -ge $NEED_GB ] || die "$n: ${f% *} GB free at $MODEL (${f#* } GB already there); the checkpoint needs $NEED_GB GB. Set MODEL to a bigger disk"
  done
  # a token for a gated repository travels on the script's standard input, not on a command line
  for n in "${targets[@]}"; do { [ -z "${HF_TOKEN:-}" ] || echo "import os; os.environ['HF_TOKEN'] = '$HF_TOKEN'"; cat $HERE/fetch-weights.py; } | rsh $n "python3 - '$WEIGHTS' '$MODEL' ${VERIFY:-0}" > $LOG/weights-$n.log 2>&1 & done
  echo "  downloading on ${#targets[@]} Spark(s); progress: tail -f $LOG/weights-<address>.log"
  while [ -n "$(jobs -rp)" ]; do sleep 60; for n in "${targets[@]}"; do echo "  $n: $(tail -1 $LOG/weights-$n.log | cut -c1-110)"; done; done; wait
  for n in "${targets[@]}"; do grep -q "^WEIGHTS OK" $LOG/weights-$n.log || die "$n: the download did not finish ($(tail -1 $LOG/weights-$n.log)); run install.sh again to resume"; done
fi
for n in "${N[@]}"; do rsh $n "test -f '$MODEL/config.json' && test -f '$MODEL/model.safetensors.index.json'" || die "$n does not see $MODEL (the same path is needed on all four)"; done

say "5/5 mlx-serve/my-cluster.env and the start check"
cat > $HERE/my-cluster.env <<ENV
# written by mlx-serve/install.sh on $(date +%F); one-shot.sh, build-engine.sh and install.sh read it
export NODES="${N[*]}"
export MODEL=$MODEL
export IMAGE=$IMAGE
ENV
env -u DRAFT -u DRAFTER -u PRESET -u CONTEXT -u PARALLEL NODES="${N[*]}" MODEL="$MODEL" IMAGE="$IMAGE" $HERE/one-shot.sh check || die "the check failed: fix the lines above (a rank needs about 100 GB of free memory)"
echo "READY. Start: mlx-serve/one-shot.sh up    (32,768 tokens x 4 streams; PRESET=250k for 250,000 x 2, PRESET=1m for 1,000,000 x 1)"
echo "       Then:  mlx-serve/one-shot.sh bench  |  HOST=${N[0]} python3 mlx-serve/checks.py  |  mlx-serve/one-shot.sh down"
