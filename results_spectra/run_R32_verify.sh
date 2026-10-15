#!/bin/bash
# R32 step 9 checks (user: "verify your assumptions"): generic-K irreducibility of the block-first
# pieces at the production sizes.  Runs after run_R32_blockfirst.sh; one job at a time.
HERE="$(cd "$(dirname "$0")" && pwd)"; LOG="$HERE/runlog_v2.txt"
cd "$HERE/../code"
# (production queue finished before this was started; no wait -- a pgrep wait matched its own launcher)
echo "=== start verify $(date -Is)" >> "$LOG"
for job in "vtower 153 15" "vring 57 16 1" "vring 153 16 1" "vring 153 16 2" "vring 54 16 1" "vring 54 16 2" "vring 147 16 1" "vring 60 16 1" "vring 60 16 3" "vtower 60 16"; do
  set -- $job
  t0=$(date +%s)
  OMP_NUM_THREADS=12 ~/pyenvs/main/bin/python -m qca_fragmentation.spectral.blockfirst $1 $2 $3 ${4:-1} >> "$LOG" 2>&1
  echo "$(date -Is) verify $job wall=$(( $(date +%s) - t0 ))s rc=$?" >> "$LOG"
done
echo "=== done verify $(date -Is)" >> "$LOG"
