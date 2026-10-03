#!/bin/bash
# One-shot: full GLM-5.3 on four DGX Sparks with TensorFold (TP=4), from the prebuilt image, launched from any machine
# that can ssh to the four Sparks. The image carries TensorFold's glm_moe_dsa engine with its CUDA extensions prebuilt
# and the fastest settings as defaults (NCCL Simple protocol with 2 channels, overlapped prompt all-reduces, RoCE
# one-shot reductions for decode windows, MTP drafts; DFlash2 drafts when DRAFT is set).
#
#   NODES="spark1 spark2 spark3 spark4"   fabric addresses (ConnectX-7 RoCE port), rank 0 first: it serves HTTP
#   MODEL=/models/GLM-5.3-EXL3-2.75BPW   the checkpoint, at the same path on all four (an NFS export works)
#   DRAFT=/models/GLM-5.3-DFlash2        optional: DFlash2 draft directory (same path on all four)
#   CONTEXT=36864  PORT=8890  IF=enp1s0f1np1  HCA=rocep1s0f1                    optional
#   GIDS="3 3 3 3"   optional: RoCE v2 GID index a rank (default: looked up on each node - they can move on reboot)
#   PARALLEL=4       optional: concurrent streams (each holds its own cache: use CONTEXT=32768 with 4)
#   DOCKER_ENV="-e TF_EXL3_PROMPT_DET=slots16"   optional extra env (slots16: reproducible long prompts, ~5% slower)
#
#   ./one-shot.sh up | down | logs [RANK]
# Before the first run, on every node: node/gb10-node-settings.sh (vm.compaction_proactiveness=0).
set -u
IMAGE=${IMAGE:-ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-03}
: "${NODES:?set NODES to the fabric addresses of the four Sparks, rank 0 first}"
NODES=($NODES)
IF=${IF:-enp1s0f1np1}
HCA=${HCA:-rocep1s0f1}
gid_of() {  # the RoCE v2 GID of the node's fabric IPv4 (::ffff:a.b.c.d) on $HCA port 1
  ssh "$1" "ip=\$(ip -4 -o addr show $IF | awk '{print \$4}' | cut -d/ -f1); h=\$(printf '%02x%02x:%02x%02x' \$(echo \$ip | tr . ' '));
    p=/sys/class/infiniband/$HCA/ports/1; for i in \$(ls \$p/gids); do [ \"\$(cat \$p/gid_attrs/types/\$i 2>/dev/null)\" = 'RoCE v2' ] &&
    grep -q \"ffff:\$h\$\" \$p/gids/\$i && { echo \$i; break; }; done"
}
if [ -n "${GIDS:-}" ]; then GIDS=($GIDS); else
  GIDS=(); for n in "${NODES[@]}"; do g=$(gid_of "$n"); [ -n "$g" ] || { echo "no RoCE v2 IPv4 GID on $n ($HCA)"; exit 1; }; GIDS+=("$g"); done
fi
MASTER=${MASTER:-${NODES[0]}}
CONTEXT=${CONTEXT:-36864}
PORT=${PORT:-8890}
DRAFT=${DRAFT:-}
PARALLEL=${PARALLEL:-1}
DOCKER_ENV=${DOCKER_ENV:-}
NAME=tf-glm53

up() {
  : "${MODEL:?set MODEL to the checkpoint directory (same path on all four nodes)}"
  local mounts="-v $MODEL:$MODEL:ro" denv=""
  if [ -n "$DRAFT" ]; then mounts="$mounts -v $DRAFT:$DRAFT:ro"; denv="-e TF_GLM53_DFLASH=$DRAFT"; fi
  for r in 3 2 1 0; do
    ssh "${NODES[$r]}" "docker rm -f $NAME-r$r >/dev/null 2>&1; docker run -d --name $NAME-r$r --gpus all \
      --network host --ipc=host --device=/dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK $mounts $denv \
      -e NCCL_SOCKET_IFNAME=$IF -e NCCL_IB_HCA=$HCA -e NCCL_IB_GID_INDEX=${GIDS[$r]} $DOCKER_ENV \
      $IMAGE python3 -m tensorfold.cli serve $MODEL --tp 4 --rank $r --master $MASTER --port $PORT \
      --host 0.0.0.0 --name glm-5.3-tf --context $CONTEXT --parallel $PARALLEL" >/dev/null && echo "rank $r started on ${NODES[$r]} (GID ${GIDS[$r]})"
  done
  echo "loading (~10 min from NFS, then decode graphs are captured); then: http://${NODES[0]}:$PORT/v1"
}

case "${1:-}" in
  up) up ;;
  down) for r in 0 1 2 3; do ssh "${NODES[$r]}" "docker rm -f $NAME-r$r >/dev/null 2>&1"; done; echo stopped ;;
  logs) r=${2:-0}; ssh "${NODES[$r]}" "docker logs -f $NAME-r$r" ;;
  *) sed -n 2,18p "$0"; exit 1 ;;
esac
