#!/bin/bash
# Run on every DGX Spark of the cluster before serving (runtime settings; they do not survive a reboot).
set -e
# Proactive memory compaction migrates pages under the GB10 GPU and stalls a rank ~130 ms at a time; in a
# tensor-parallel job every other rank waits for it at the next collective (a quarter of decode rounds took 200 ms
# instead of 70 before this).
sudo sysctl -w vm.compaction_proactiveness=0
# Check: no bulk NFS / copies over the RoCE port while serving (they add decode latency).
