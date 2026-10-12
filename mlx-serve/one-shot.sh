#!/bin/bash
# Full GLM-5.3 EXL3 2.75 bpw on four DGX Sparks: mlx-serve on rank 0 with TensorFold 1.0.7's GLM-5.3 CUDA
# inference engine at tensor parallel 4, ranks 1..3 following. One container a Spark, started over ssh from any machine.
#   ./one-shot.sh check | up | down | status | bench
# Settings (environment, or `source cluster.env` first):
#   NODES    the four Sparks' fabric addresses, rank 0 first (required)
#   MODEL    the checkpoint directory, the same path on all four (required)
#   IMAGE    [ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-12-r2-mlx-serve]
#   DRAFTER  dspark (the DSpark ft2 drafter in the image, default) | mtp (GLM-5.3's own MTP layer) |
#            dflash (DRAFT=<a DFlash2 directory on all four>; CC BY-NC-ND, you download it yourself)
#   PRESET   unset: 32,768 tokens x 4 streams | 64k: 65,536 x 2 | 250k: 250,000 x 2 | 1m: 1,000,000 x 1 (one request
#            at a time). 250k and 1m split the cache across the four Sparks.
#   CONTEXT  PARALLEL  (default to the preset's; another shape needs its own kernel pack)
#   PORT [18992]  RAILS [2]  MEM [110g]  RDV [29882] (rank rendezvous port)
#   HEAD_Q   unset: the 6-bit output head when $MODEL/lm_head-6bit/lm_head-6bit.safetensors exists (about 3 % faster
#            decode, KL +0.0001), else the bf16 head | off: the bf16 head
#   DOCKER_ARGS  extra `docker run` arguments for every rank (e.g. -v a kernel pack for another context)
set -u
read -r -a NODES <<<"${NODES:-}"
[ ${#NODES[@]} = 4 ] || { echo "NODES must list four fabric addresses, rank 0 first"; exit 2; }
MODEL=${MODEL:-}; DRAFT=${DRAFT:-}; DRAFTER=${DRAFTER:-dspark}
IMAGE=${IMAGE:-ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-12-r2-mlx-serve}
PRESET=${PRESET:-}
case "$PRESET" in "") c=32768 p=4 ;; 64k) c=65536 p=2 ;; 250k) c=250000 p=2 ;; 1m) c=1000000 p=1 ;;
  *) echo "PRESET=$PRESET (64k, 250k or 1m)"; exit 2 ;; esac
CONTEXT=${CONTEXT:-$c}; PARALLEL=${PARALLEL:-$p}; PORT=${PORT:-18992}; RAILS=${RAILS:-2}; MEM=${MEM:-110g}; RDV=${RDV:-29882}
HERE=$(cd "$(dirname "$0")" && pwd); RUN=${RUN:-$HERE/run}
rsh() { ssh -o ConnectTimeout=10 -o BatchMode=yes "$@"; }
say() { echo "$(date +%H:%M:%S) $*" | tee -a $RUN/events.log; }
down() { for n in "${NODES[@]}"; do rsh $n 'docker ps -aq --filter label=keys-glm53 | xargs -r docker rm -f' >/dev/null 2>&1 & done; wait; }
avail() { rsh $1 "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo"; }

check() {   # read-only; exits 1 if anything blocks a start
  local bad=0 r n out m sw
  for r in 0 1 2 3; do n=${NODES[$r]}
    out=$(rsh "$n" "
      docker image inspect $IMAGE >/dev/null 2>&1 && echo image=ok || echo image=missing
      [ -f '${MODEL:-/nonexistent}/config.json' ] && echo model=ok || echo model=missing
      [ '$DRAFTER' != dflash ] || { [ -f '$DRAFT/config.json' ] && echo draft=ok || echo draft=missing; }
      echo compaction=\$(cat /proc/sys/vm/compaction_proactiveness) swappiness=\$(cat /proc/sys/vm/swappiness)
      sudo -n true 2>/dev/null && echo sudo=ok || echo sudo=no
      echo swap_used_gb=\$(awk '/SwapTotal/{t=\$2} /SwapFree/{f=\$2} END{print int((t-f)/1048576)}' /proc/meminfo)
      echo memfree_gb=\$(awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo)
      echo gpu_procs=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)" 2>&1) || { echo "FAIL rank $r ($n): ssh (BatchMode) failed"; bad=1; continue; }
    echo "rank $r ($n): $(tr '\n' ' ' <<<"$out")"
    grep -q image=missing <<<"$out" && { echo "  FAIL image not pulled: ssh $n docker pull $IMAGE"; bad=1; }
    grep -q model=missing <<<"$out" && { echo "  FAIL MODEL not found on $n (same path on all four)"; bad=1; }
    grep -q draft=missing <<<"$out" && { echo "  FAIL DRAFT not found on $n"; bad=1; }
    grep -q "compaction=0" <<<"$out" || echo "  WARN vm.compaction_proactiveness is not 0: run node/gb10-node-settings.sh (decode stalls otherwise)"
    sw=$(grep -o 'swappiness=[0-9]*' <<<"$out" | cut -d= -f2); [ "${sw:-60}" -le 10 ] || { echo "  FAIL vm.swappiness=$sw: run node/gb10-node-settings.sh (the engine gets swapped out while loading)"; bad=1; }
    grep -q "sudo=no" <<<"$out" && echo "  WARN no passwordless sudo on $n: 'up' cannot drop page cache while the ranks load - run 'sudo sysctl vm.drop_caches=1' there during the load"
    m=$(grep -o 'memfree_gb=[0-9]*' <<<"$out" | cut -d= -f2); [ "${m:-0}" -ge 100 ] || { echo "  FAIL only ${m} GB available on $n (a rank needs ~100 GB): stop other jobs, drop page cache"; bad=1; }
    grep -q "gpu_procs=0" <<<"$out" || echo "  WARN other GPU processes on $n: every rank waits for the slowest - stop them"
  done
  [ $bad = 0 ] && echo "check: OK" || { echo "check: FAILED"; return 1; }
}

case "${1:-}" in
  down) mkdir -p $RUN; touch $RUN/STOP; down; say "down"; exit 0 ;;
  status) for r in 0 1 2 3; do echo "rank $r ${NODES[$r]}: $(rsh ${NODES[$r]} "docker ps --filter label=keys-glm53 --format '{{.Names}} {{.Status}}'" | paste -sd' ') $(avail ${NODES[$r]}) GB available"; done; exit 0 ;;
  check) check; exit $? ;;
  bench) exec python3 $HERE/bench.py http://${NODES[0]}:$PORT ;;
  up) ;;
  *) sed -n 2,14p "$0"; exit 2 ;;
esac
[ -n "$MODEL" ] || { echo "MODEL is not set"; exit 2; }
rm -rf $RUN; mkdir -p $RUN; down

# every RoCE v2 rail of a node: "subnet device gid-index order interface", the node's own subnet first
rail_scan='for d in /sys/class/net/*; do n=${d##*/}; [ -d $d/device/infiniband ] || continue
  [ "$(cat $d/operstate 2>/dev/null)" = up ] || continue
  a=$(ip -4 -o addr show dev $n | awk "{print \$4}" | head -1); [ -n "$a" ] || continue
  net=$(ip -4 -o route show dev $n scope link proto kernel | awk "{print \$1}" | head -1)
  hca=$(ls $d/device/infiniband | head -1); h=$(printf "%02x%02x:%02x%02x" $(echo ${a%/*} | tr . " "))
  p=/sys/class/infiniband/$hca/ports/1; g=
  for i in $(ls $p/gids | sort -n); do [ "$(cat $p/gid_attrs/types/$i 2>/dev/null)" = "RoCE v2" ] &&
    grep -q "ffff:$h\$" $p/gids/$i && { g=$i; break; }; done
  [ -n "$g" ] || continue
  o=$(ibv_devices 2>/dev/null | awk "NR>2{print \$1}" | grep -nx "$hca" | cut -d: -f1)
  echo "${net:-?} $hca $g ${o:-0} $n"; done'
declare -a HCAS RGIDS NGID ZHCAS ZGID IFN; XNIC=0; ref=""
for r in 0 1 2 3; do
  lines=$(rsh "${NODES[$r]}" "$rail_scan" | sort) || { say "ssh to ${NODES[$r]} failed"; exit 1; }
  own=$(awk -v ip="${NODES[$r]}" '{split($1, c, "[./]"); split(ip, d, "."); if (c[1]==d[1] && c[2]==d[2] && c[3]==d[3]) print}' <<<"$lines")
  [ -n "$own" ] || { say "${NODES[$r]}: no RoCE v2 device on its own subnet (is NODES the fabric address?)"; exit 1; }
  lines=$( { echo "$own"; grep -vxF "$own" <<<"$lines"; } | grep . | head -n "$RAILS")
  nets=$(awk '{print $1}' <<<"$lines" | paste -sd' '); [ -z "$ref" ] && ref=$nets
  [ "$nets" = "$ref" ] || { say "rank $r rails on [$nets], rank 0 on [$ref]: fix the second rail's addressing or RAILS=1"; exit 1; }
  HCAS[$r]=$(awk '{print $2}' <<<"$lines" | paste -sd,); RGIDS[$r]=$(awk '{print $3}' <<<"$lines" | paste -sd,)
  NGID[$r]=$(awk '{print $3}' <<<"$lines" | sort -u | awk 'END{if (NR==1) print}'); IFN[$r]=$(awk 'NR==1{print $5}' <<<"$lines")
  [ "$(awk '{print $4}' <<<"$lines" | paste -sd' ')" = "$(awk '{print $4}' <<<"$lines" | sort -n | paste -sd' ')" ] || XNIC=1
  # the RoCE one-shot collectives take one GID index: with differing indices they run on the first rail alone
  if [ -n "${NGID[$r]}" ]; then ZHCAS[$r]=${HCAS[$r]}; ZGID[$r]=${NGID[$r]}; else ZHCAS[$r]=${HCAS[$r]%%,*}; ZGID[$r]=${RGIDS[$r]%%,*}; fi
  say "rank $r (${NODES[$r]}): ${IFN[$r]}, rails ${HCAS[$r]} GIDs ${RGIDS[$r]}; one-shot on ${ZHCAS[$r]} (GID ${ZGID[$r]})"
done
rank_env() {
  local r=$1 e="-e RANK=$1 -e MASTER=${NODES[0]}:$RDV -e CONTEXT=$CONTEXT -e PARALLEL=$PARALLEL -e PORT=$PORT -e DRAFTER=$DRAFTER -e PRESET=$PRESET ${HEAD_Q:+-e HEAD_Q=$HEAD_Q} \
    -e NCCL_SOCKET_IFNAME=${IFN[$1]} -e NCCL_IB_HCA=${HCAS[$1]} -e TF_GLM53_ROCE_HCA=${HCAS[$1]} -e TF_GLM53_ROCE_GIDS=${RGIDS[$1]} \
    -e TF_GLM53_HCAS=${ZHCAS[$1]} -e TF_GLM53_GID=${ZGID[$1]} -v $MODEL:/model:ro"
  [ $DRAFTER = dflash ] && e="$e -e DRAFT=$DRAFT -v $DRAFT:$DRAFT:ro"
  if [ -n "${NGID[$r]}" ]; then e="$e -e NCCL_IB_GID_INDEX=${NGID[$r]}"; else e="$e -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET"; fi
  if (( XNIC )); then e="$e -e NCCL_CROSS_NIC=1 -e NCCL_IB_SUBNET_AWARE_ROUTING=1"; else e="$e -e NCCL_CROSS_NIC=0"; fi
  # any TF_GLM53_* you export reaches every rank
  for v in $(compgen -v TF_GLM53_); do e="$e -e $v=${!v}"; done; echo "$e"
}
flags="--label keys-glm53=1 --user \$(id -u):\$(id -g) -e HOME=/tmp --gpus all --network host --ipc host --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK --memory $MEM --memory-swap $MEM ${DOCKER_ARGS:-}"
for r in 0 1 2 3; do a=$(avail ${NODES[$r]}); [ "${a:-0}" -ge 100 ] || { say "${NODES[$r]} has ${a} GB available (< 100): stop other jobs, drop page cache"; exit 1; }; done
# GB10's GPU allocations can only use free memory: drop clean page cache on the nodes while the ranks load
( until [ -e $RUN/STOP ] || [ -e $RUN/SERVING ]; do for n in "${NODES[@]}"; do rsh $n 'sudo -n sysctl -q -w vm.drop_caches=1' >/dev/null 2>&1; done; sleep 10; done ) &
# memory watchdog: under 3 GB available or 2 GB of swap in use on any node stops every rank
( while [ ! -e $RUN/STOP ]; do for n in "${NODES[@]}"; do
    a=$(rsh $n "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo; awk '/SwapTotal|SwapFree/{print \$2}' /proc/meminfo | paste -sd' ' | awk '{print int((\$1-\$2)/1048576)}'" | paste -sd/)
    m=${a%%/*}; s=${a##*/}; echo "$(date +%H:%M:%S) $n avail $m swap $s" >> $RUN/mem.log
    if [ "${m:-99}" -lt 3 ] || [ "${s:-0}" -gt 2 ]; then say "MEMORY ABORT $n: avail ${m} GB swap ${s} GB - stopping every rank"; touch $RUN/STOP $RUN/ABORTED
      r=0; for x in "${NODES[@]}"; do rsh $x "docker logs keys-glm53-r$r" > $RUN/abort-r$r.log 2>&1; r=$((r+1)); done; down; fi
  done; sleep 10; done ) &
for r in 1 2 3 0; do
  rsh ${NODES[$r]} "docker run -d --name keys-glm53-r$r $flags $(rank_env $r) $IMAGE" >> $RUN/start.log 2>&1 || { say "rank $r did not start (run/start.log)"; touch $RUN/STOP; down; exit 1; }
done
say "started: rank 0 on ${NODES[0]}:$PORT, ranks 1..3 following; model $MODEL, drafter $DRAFTER $DRAFT, context $CONTEXT, parallel $PARALLEL"
t0=$(date +%s)
while :; do
  [ -e $RUN/STOP ] && { say "stopped while loading"; exit 1; }
  if curl -s -m 3 http://${NODES[0]}:$PORT/health >/dev/null 2>&1 && rsh ${NODES[0]} "docker logs keys-glm53-r0 2>&1 | grep -q 'engine ready'"; then break; fi
  for r in 0 1 2 3; do rsh ${NODES[$r]} "docker ps -q --filter name=keys-glm53-r$r | grep -q ." || { say "rank $r exited: $(rsh ${NODES[$r]} "docker logs --tail 4 keys-glm53-r$r 2>&1" | tr '\n' ' ' | cut -c1-400)"; touch $RUN/STOP; for x in 0 1 2 3; do rsh ${NODES[$x]} "docker logs keys-glm53-r$x" > $RUN/fail-r$x.log 2>&1; done; down; exit 1; }; done
  (( $(date +%s) - t0 > 2400 )) && { say "not up after 40 min"; touch $RUN/STOP; down; exit 1; }
  sleep 15
done
touch $RUN/SERVING; say "SERVING after $(( $(date +%s) - t0 )) s: http://${NODES[0]}:$PORT/v1 (model name: model)"
for r in 0 1 2 3; do say "rank $r: $(avail ${NODES[$r]}) GB available"; done
say "the memory watchdog keeps running in the background; './one-shot.sh down' stops it and the ranks"
