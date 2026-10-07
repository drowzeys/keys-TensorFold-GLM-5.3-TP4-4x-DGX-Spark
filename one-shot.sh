#!/bin/bash
# One-shot: full GLM-5.3 on four DGX Sparks with TensorFold (TP=4) from the prebuilt image, launched from any machine
# that can ssh to the four Sparks. The image's defaults are the fastest measured settings (see README.md); nothing
# here needs tuning. Agents: follow AGENTS.md, which runs `check`, `up`, `wait` and `bench` in order.
#
#   NODES="spark1 spark2 spark3 spark4"   fabric addresses (ConnectX-7 RoCE port), rank 0 first: it serves HTTP
#   MODEL=/models/GLM-5.3-EXL3-2.75BPW   the checkpoint, at the same path on all four (an NFS export works)
#   optional:
#   DSPARK=/models/keys-GLM-5.3-speculator.dspark-ft2   DSpark drafter (GLM-5.3 license; hf download
#                                        drowzeys/keys-GLM-5.3-speculator.dspark-ft2): drafts alone, MTP not loaded,
#                                        confidence policy 0.3 - +5 % prose / +12 % code at 32K, one stream (--parallel 1)
#   DRAFT=/models/GLM-5.3-DFlash2        DFlash2 drafter (incoai, CC BY-NC-ND 4.0: you download it; +10 % on code)
#   CONTEXT=140000                       window a stream (1000000 turns on decode context parallelism)
#   PARALLEL=1                           concurrent streams (each holds its own cache: PARALLEL=4 needs CONTEXT=32768)
#   RAILS=2                              RoCE rails: both PCIe twins of the QSFP port (1 = the NODES subnet's only)
#   PORT=8890  IF=enp1s0f1np1  IMAGE=...  DOCKER_ENV="-e KEY=VALUE ..." (extra engine settings, e.g.
#                                         -e TF_EXL3_PROMPT_DET=slots16 for reproducible long prompts, ~5 % slower prefill)
#   SERVE_ARGS="--kv-dtype int8"           extra serve flags
#
#   ./one-shot.sh check        read-only: ssh, docker, image, checkpoint, rails / GIDs, node settings, free memory
#   ./one-shot.sh up           start the four ranks          ./one-shot.sh wait    until rank 0 answers (~10 min; run it
#                                                            right after up: it drops page cache on the nodes while they load)
#   ./one-shot.sh bench        short decode + 32K prefill    ./one-shot.sh down | logs [RANK]
set -u
IMAGE=${IMAGE:-ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-07}
: "${NODES:?set NODES to the fabric addresses of the four Sparks, rank 0 first}"
NODES=($NODES)
[ ${#NODES[@]} -eq 4 ] || { echo "NODES must list exactly four Sparks (rank 0 first)"; exit 1; }
MASTER=${MASTER:-${NODES[0]}}
IF=${IF:-enp1s0f1np1}
RAILS=${RAILS:-2}
CONTEXT=${CONTEXT:-140000}
PORT=${PORT:-8890}
DRAFT=${DRAFT:-}
DSPARK=${DSPARK:-}
PARALLEL=${PARALLEL:-1}
DOCKER_ENV=${DOCKER_ENV:-}
SERVE_ARGS=${SERVE_ARGS:-}
NAME=tf-glm53
URL="http://${NODES[0]}:$PORT"

# Rails. A DGX Spark's QSFP port is two PCIe x4 RoCE devices ("twins", ~112 Gb/s each). On each node: every up RoCE
# netdev with an IPv4 address and a RoCE v2 GID for it, ordered by subnet (NODES[r]'s first), so rail i is one subnet
# on every node. Devices are matched by subnet, never by name (enumeration can swap across reboots), and GID indices
# are looked up at every start (they move across reboots).
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
declare -a HCAS=() RGIDS=() NGID=()
XNIC=0
detect_rails() {
  local ref="" r lines own nets
  for r in 0 1 2 3; do
    lines=$(ssh -o ConnectTimeout=10 "${NODES[$r]}" "$rail_scan" | sort) || { echo "ssh to ${NODES[$r]} failed"; return 1; }
    own=$(awk -v ip="${NODES[$r]}" '{split($1, c, "[./]"); split(ip, d, "."); if (c[1]==d[1] && c[2]==d[2] && c[3]==d[3]) print}' <<<"$lines")
    [ -n "$own" ] || { echo "${NODES[$r]}: no RoCE v2 device on its own subnet (is NODES the fabric address?)"; return 1; }
    lines=$( { echo "$own"; grep -vxF "$own" <<<"$lines"; } | grep . | head -n "$RAILS")
    nets=$(awk '{print $1}' <<<"$lines" | paste -sd' ')
    [ -z "$ref" ] && ref=$nets
    if [ "$nets" != "$ref" ]; then
      echo "rank $r rails on [$nets], rank 0 on [$ref]: subnets differ - fix the second rail's addressing or RAILS=1"; return 1
    fi
    HCAS[$r]=$(awk '{print $2}' <<<"$lines" | paste -sd,)
    RGIDS[$r]=$(awk '{print $3}' <<<"$lines" | paste -sd,)
    NGID[$r]=$(awk '{print $3}' <<<"$lines" | sort -u | awk 'END{if (NR==1) print}')
    [ "$(awk '{print $4}' <<<"$lines" | paste -sd' ')" = "$(awk '{print $4}' <<<"$lines" | sort -n | paste -sd' ')" ] || XNIC=1
    echo "rank $r (${NODES[$r]}): rails ${HCAS[$r]} on [$nets], GIDs ${RGIDS[$r]}"
  done
}
nccl_env() {
  local r=$1 e="-e NCCL_SOCKET_IFNAME=$IF -e NCCL_IB_HCA=${HCAS[$1]} -e TF_GLM53_ROCE_HCA=${HCAS[$1]} -e TF_GLM53_ROCE_GIDS=${RGIDS[$1]}"
  if [ -n "${NGID[$r]}" ]; then e="$e -e NCCL_IB_GID_INDEX=${NGID[$r]}"
  else e="$e -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET"; fi
  if (( XNIC )); then e="$e -e NCCL_CROSS_NIC=1 -e NCCL_IB_SUBNET_AWARE_ROUTING=1"; else e="$e -e NCCL_CROSS_NIC=0"; fi
  echo "$e"
}

check() {   # read-only; prints FAIL lines and exits 1 if anything blocks a start
  local bad=0 r n
  for r in 0 1 2 3; do n=${NODES[$r]}
    out=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$n" "
      docker image inspect $IMAGE >/dev/null 2>&1 && echo image=ok || echo image=missing
      [ -f '${MODEL:-/nonexistent}/config.json' ] && echo model=ok || echo model=missing
      [ -z '$DRAFT' ] || { [ -f '$DRAFT/config.json' ] && echo draft=ok || echo draft=missing; }
      [ -z '$DSPARK' ] || { [ -f '$DSPARK/config.json' ] && echo draft=ok || echo draft=missing; }
      echo compaction=\$(cat /proc/sys/vm/compaction_proactiveness)
      echo swappiness=\$(cat /proc/sys/vm/swappiness)
      sudo -n true 2>/dev/null && echo sudo=ok || echo sudo=no
      echo swap_used_gb=\$(awk '/SwapTotal/{t=\$2} /SwapFree/{f=\$2} END{print int((t-f)/1048576)}' /proc/meminfo)
      echo memfree_gb=\$(awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo)
      echo gpu_procs=\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
      echo containers=\$(docker ps -q | grep -c .)
      getent hosts ghcr.io >/dev/null && echo dns=ok || echo dns=fail" 2>&1) || { echo "FAIL rank $r ($n): ssh (BatchMode) failed"; bad=1; continue; }
    echo "rank $r ($n): $(tr '\n' ' ' <<<"$out")"
    grep -q image=missing <<<"$out" && { echo "  FAIL image not pulled: ssh $n docker pull $IMAGE"; bad=1; }
    grep -q dns=fail <<<"$out" && echo "  WARN $n cannot resolve ghcr.io (no DNS server set after a reboot?): sudo resolvectl dns <its default-route interface> 1.1.1.1"
    grep -q model=missing <<<"$out" && { echo "  FAIL MODEL not found on $n (same path on all four)"; bad=1; }
    grep -q draft=missing <<<"$out" && { echo "  FAIL DRAFT not found on $n"; bad=1; }
    grep -q "compaction=0" <<<"$out" || echo "  WARN vm.compaction_proactiveness is not 0: run node/gb10-node-settings.sh (decode stalls otherwise)"
    sw=$(grep -o 'swappiness=[0-9]*' <<<"$out" | cut -d= -f2); [ "${sw:-60}" -le 10 ] || { echo "  FAIL vm.swappiness=$sw: run node/gb10-node-settings.sh (the engine gets swapped out while loading)"; bad=1; }
    grep -q "sudo=no" <<<"$out" && echo "  WARN no passwordless sudo on $n: './one-shot.sh wait' cannot drop page cache while loading - run 'sync; sudo sysctl vm.drop_caches=1' on every node during the load, or the engine may refuse to start"
    su=$(grep -o 'swap_used_gb=[0-9]*' <<<"$out" | cut -d= -f2); [ "${su:-0}" -le 1 ] || echo "  WARN ${su} GB of swap in use on $n: sudo swapoff -a && sudo swapon -a before starting"
    m=$(grep -o 'memfree_gb=[0-9]*' <<<"$out" | cut -d= -f2); [ "${m:-0}" -ge 100 ] || echo "  WARN only ${m} GB available (the engine needs ~100 GB a node): stop other jobs, drop page cache"
    grep -q "gpu_procs=0" <<<"$out" || echo "  WARN other GPU processes on $n: every rank waits for the slowest - stop them"
  done
  detect_rails || bad=1
  (( XNIC )) && echo "note: ibverbs order differs from subnet order on some node: NCCL crosses NICs by subnet"
  [ $bad = 0 ] && echo "check: OK" || { echo "check: FAILED"; return 1; }
}

up() {
  : "${MODEL:?set MODEL to the checkpoint directory (same path on all four nodes)}"
  detect_rails || exit 1
  local mounts="-v $MODEL:$MODEL:ro" denv="" r n
  # a clean start: page cache dropped (an image pull or unpack leaves tens of GB that GB10's GPU allocations cannot use)
  for n in "${NODES[@]}"; do ssh "$n" 'sync; sudo -n sysctl -q -w vm.drop_caches=3' >/dev/null 2>&1 & done; wait
  # a lane stopped seconds ago may not have handed its GPU memory back yet: starting then left a node with ~1 GB free
  # after warm-up (2026-10-05). Wait until every node is back near full (up to 2 min).
  local i ok
  for i in $(seq 1 24); do ok=1
    for n in "${NODES[@]}"; do
      a=$(ssh "$n" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null); [ "${a:-0}" -ge 100 ] || ok=0
    done; [ $ok = 1 ] && break; sleep 5; done
  [ $ok = 1 ] || echo "warning: a node still has under 100 GB available; another job may be using its memory (./one-shot.sh check)"
  if [ -n "$DRAFT" ]; then mounts="$mounts -v $DRAFT:$DRAFT:ro"; denv="-e TF_GLM53_DFLASH=$DRAFT"; fi
  if [ -n "$DSPARK" ]; then                       # DSpark alone: MTP not loaded, confidence policy (2026-10-07 A/B)
    [ "$PARALLEL" = 1 ] || { echo "DSPARK drafts one stream at a time: PARALLEL=1"; exit 1; }
    mounts="$mounts -v $DSPARK:$DSPARK:ro"
    denv="$denv -e TF_GLM53_DSPARK=$DSPARK -e TF_GLM53_DSPARK_POLICY=confidence -e TF_GLM53_DSPARK_CONFIDENCE=0.3"
    SERVE_ARGS="--mtp-drafts 0 $SERVE_ARGS"
  fi
  for r in 3 2 1 0; do
    ssh "${NODES[$r]}" "docker rm -f $NAME-r$r >/dev/null 2>&1; docker run -d --name $NAME-r$r --gpus all \
      --network host --ipc=host --device=/dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK $mounts $denv \
      $(nccl_env $r) $DOCKER_ENV \
      $IMAGE python3 -m tensorfold.cli serve $MODEL --tp 4 --rank $r --master $MASTER --port $PORT \
      --host 0.0.0.0 --name glm-5.3-tf --context $CONTEXT --parallel $PARALLEL $SERVE_ARGS" >/dev/null && echo "rank $r started on ${NODES[$r]}"
  done
  echo "loading (~10 min from NFS, then decode graphs are captured): ./one-shot.sh wait"
}

wait_up() {
  # While the ranks load, drop clean page cache on every node every 15 s: GB10's GPU allocations count only free
  # memory, and the checkpoint streaming through the page cache (above all on the node exporting it over NFS, whose
  # server-side cache the ranks cannot drop) otherwise leaves the engine short at warm-up. Clean pages only: safe.
  local i
  for i in $(seq 1 120); do
    for n in "${NODES[@]}"; do ssh -o ConnectTimeout=5 "$n" 'sync; sudo -n sysctl -q -w vm.drop_caches=1' >/dev/null 2>&1 & done; wait
    if curl -s -m 5 -o /dev/null -w '%{http_code}' "$URL/v1/models" | grep -q 200; then
      echo "up: $URL/v1"
      for n in "${NODES[@]}"; do
        a=$(ssh "$n" "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null)
        echo "  $n: ${a} GB available"; [ "${a:-0}" -ge 6 ] || echo "  WARN $n has under 6 GB available while serving: ./one-shot.sh down, wait a minute, up again"
      done
      return 0
    fi
    for r in 0 1 2 3; do
      ssh "${NODES[$r]}" "docker ps -q -f name=$NAME-r$r" | grep -q . ||
        { echo "rank $r exited:"; ssh "${NODES[$r]}" "docker logs --tail 25 $NAME-r$r 2>&1"; return 1; }
    done
    sleep 15
  done
  echo "not up after 30 min: ./one-shot.sh logs 0"; return 1
}

bench() {   # short greedy prose + code decode, then one cold 32K prompt (prefill / TTFT)
  python3 - "$URL" <<'EOF'
import json, sys, time, urllib.request
U = sys.argv[1]
def ask(content, n, **kw):
    body = dict(model="glm-5.3-tf", messages=[dict(role="user", content=content)], max_tokens=n, temperature=0, **kw)
    t = time.time()
    b = json.load(urllib.request.urlopen(urllib.request.Request(U + "/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"}), timeout=1800))
    return b, time.time() - t
for name, p in (("prose", "Write a detailed explanation of how a hash table works, including collisions and resizing."),
                ("code", "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.")):
    b, dt = ask(p, 300)
    print(f"{name:6s} {b['usage']['completion_tokens'] / dt:6.1f} tok/s (300 tokens, greedy, thinking on, request time)", flush=True)
filler = "\n".join(f"Entry {i}: shelf {i % 97} holds volume {i * 31 % 1009} of the river survey." for i in range(1450))
lines = filler.split("\n"); lines.insert(len(lines) // 2, "The passphrase is CRIMSON-OTTER-7731.")
b, dt = ask("\n".join(lines) + "\n\nWhat is the passphrase? Reply with it only.", 4096)
n = b["usage"]["prompt_tokens"]
print(f"prefill + answer {n} tokens: {dt:.1f} s; needle {'PASS' if 'CRIMSON-OTTER-7731' in (b['choices'][0]['message'].get('content') or '') else 'FAIL'}")
EOF
}

case "${1:-}" in
  check) check ;;
  up) up ;;
  wait) wait_up ;;
  bench) bench ;;
  down) for r in 0 1 2 3; do ssh "${NODES[$r]}" "docker rm -f $NAME-r$r >/dev/null 2>&1"; done; echo stopped ;;
  logs) r=${2:-0}; ssh "${NODES[$r]}" "docker logs -f $NAME-r$r" ;;
  *) sed -n 2,22p "$0"; exit 1 ;;
esac
