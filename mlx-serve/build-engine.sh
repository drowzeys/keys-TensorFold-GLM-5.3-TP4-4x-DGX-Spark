#!/bin/bash
# Build the GLM-5.3 engine of the image from its published source, on every Spark, and keep it beside the image.
#   mlx-serve/build-engine.sh            # on the four NODES (ssh), one build a node at the same time
#   mlx-serve/build-engine.sh --local    # on this machine only (no GPU needed: it compiles)
# then:  ENGINE=source mlx-serve/one-shot.sh up      # starts the ranks with that build mounted over the image's engine
# What it builds: libtensorfold_glm53.so (the engine behind rank 0's server) and tensorfold-native (ranks 1..3), with
# their CUDA kernel packs, from https://github.com/drowzeys/TensorFold at TF_COMMIT (branch glm53-tp4-2026-10-12-r2:
# TensorFold 1.0.7 by Ash Hart and the TensorFold contributors, plus our unofficial GLM-5.3 port).
# What stays the image's (no public source to build here): rank 0's HTTP server (mlx-serve by David Dalcu with our
# bridge to the engine library), the precompiled Triton kernel packs and tile tables (aot*/, ref*/) and the drafter.
# Tools: the image's own CUDA 13 compiler; Zig 0.17.0 and the source are downloaded (checksum and commit pinned).
# Settings: NODES, IMAGE (as one-shot.sh), ENGINE_DIR [~/.local/share/keys-glm53/engine] (same path on every node),
#   TF_COMMIT, JOBS [8], BUILD_MEM [24g]. About 10 minutes a node and 2 GB of disk.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
[ -n "${NODES:-}" ] || [ ! -f $HERE/my-cluster.env ] || . $HERE/my-cluster.env
IMAGE=${IMAGE:-ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark:2026-10-12-r2-mlx-serve}
TF_COMMIT=${TF_COMMIT:-00c3bb11dc8b72fef0b26f2beb7a3e7f8c3a43ab}
ENGINE_DIR=${ENGINE_DIR:-.local/share/keys-glm53/engine}; JOBS=${JOBS:-8}; BUILD_MEM=${BUILD_MEM:-24g}
ZIG_URL=https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz
ZIG_SHA=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
# runs on a node; $1 = engine directory (relative to the home directory, or absolute)
node_build='set -eu
case "$1" in /*) E=$1 ;; *) E=$HOME/$1 ;; esac
if [ "$(cat $E/commit 2>/dev/null)" = "$TF_COMMIT" ] && [ -x $E/zig-out/native/bin/tensorfold-native ] && [ -s $E/zig-out/lib/libtensorfold_glm53.so ]; then
  echo "engine already built at ${TF_COMMIT:0:8} in $E"; exit 0; fi
[ "$(uname -m)" = aarch64 ] || { echo "this builds on a DGX Spark (aarch64); found $(uname -m)"; exit 1; }
free=$(df --output=avail -BG "$(dirname "$E")" 2>/dev/null | tail -1 | tr -dc 0-9); mkdir -p $E
free=$(df --output=avail -BG $E | tail -1 | tr -dc 0-9); [ "${free:-0}" -ge 4 ] || { echo "only ${free} GB of disk free at $E (4 needed)"; exit 1; }
docker image inspect $IMAGE >/dev/null 2>&1 || docker pull -q $IMAGE
rm -rf $E/src $E/zig-out $E/commit
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp --memory $BUILD_MEM --memory-swap $BUILD_MEM -v $E:/e -w /e \
  -e TF_COMMIT=$TF_COMMIT -e ZIG_URL=$ZIG_URL -e ZIG_SHA=$ZIG_SHA -e JOBS=$JOBS --entrypoint bash $IMAGE -c "
  set -eu
  if [ ! -x zig/zig ]; then curl -fsSL --retry 3 -o zig.tar.xz \$ZIG_URL; echo \"\$ZIG_SHA  zig.tar.xz\" | sha256sum -c - >/dev/null
    mkdir -p zig && tar -xJf zig.tar.xz --strip-components=1 -C zig && rm zig.tar.xz; fi
  curl -fsSL --retry 3 -o src.tar.gz https://github.com/drowzeys/TensorFold/archive/\$TF_COMMIT.tar.gz
  mkdir -p src && tar -xzf src.tar.gz --strip-components=1 -C src && rm src.tar.gz
  export PATH=/e/zig:/usr/local/cuda/bin:/usr/bin:/bin ZIG_LOCAL_CACHE_DIR=/e/cache/local ZIG_GLOBAL_CACHE_DIR=/e/cache/global
  cd src
  nice -n 19 zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=safe --prefix /e/zig-out -j\$JOBS fatbins
  nice -n 19 zig build -Dfatbins=/e/zig-out/fatbin -Dsm=121 -Doptimize=safe --prefix /e/zig-out -j\$JOBS native glm53-lib"
echo $TF_COMMIT > $E/commit; rm -rf $E/src $E/cache
docker run --rm --network none -v $E/zig-out:/z:ro --entrypoint /z/native/bin/tensorfold-native $IMAGE --version
ls -la $E/zig-out/lib/libtensorfold_glm53.so $E/zig-out/native/bin/tensorfold-native | awk "{print \$5, \$9}"'
vars="TF_COMMIT=$TF_COMMIT IMAGE=$IMAGE ZIG_URL=$ZIG_URL ZIG_SHA=$ZIG_SHA JOBS=$JOBS BUILD_MEM=$BUILD_MEM"
if [ "${1:-}" = --local ]; then env $vars bash -c "$node_build" build "$ENGINE_DIR"; exit $?; fi
read -r -a N <<<"${NODES:-}"; [ ${#N[@]} = 4 ] || { echo "NODES must list four fabric addresses (or run install.sh first, or use --local)"; exit 2; }
LOG=${LOG:-$HERE/run-build}; mkdir -p $LOG; rc=0
for n in "${N[@]}"; do ssh -o ConnectTimeout=10 -o BatchMode=yes -o ServerAliveInterval=30 $n "$vars bash -s -- '$ENGINE_DIR'" <<<"$node_build" > $LOG/build-$n.log 2>&1 & done; wait
for n in "${N[@]}"; do echo "== $n: $(tail -3 $LOG/build-$n.log | tr '\n' ' ' | cut -c1-300)"; grep -q "tensorfold-native 1.0.7" $LOG/build-$n.log || grep -q "already built" $LOG/build-$n.log || rc=1; done
[ $rc = 0 ] && echo "build-engine: OK. Start with: ENGINE=source mlx-serve/one-shot.sh up" || { echo "build-engine: FAILED (logs in $LOG)"; exit 1; }
