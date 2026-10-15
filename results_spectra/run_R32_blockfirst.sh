#!/bin/bash
# R32 step 9 (2026-10-15): block-first spectra for the relevant cases only (user request).
# Eigenvalues only; one job at a time; log appended to runlog_v2.txt.
HERE="$(cd "$(dirname "$0")" && pwd)"; LOG="$HERE/runlog_v2.txt"
cd "$HERE/../code"
echo "=== start blockfirst $(date -Is)" >> "$LOG"
for job in "tower 153 15" "ring 57 16" "ring 54 16" "ring 153 16" "ring 147 16" "ring 60 16" "tower 60 16"; do
  set -- $job
  f="$HERE/blockfirst/${1}_N${3}_W${2}.npz"
  [ -f "$f" ] && continue
  t0=$(date +%s)
  OMP_NUM_THREADS=12 ~/pyenvs/main/bin/python -m qca_fragmentation.spectral.blockfirst $1 $2 $3 >> "$LOG" 2>&1
  echo "$(date -Is) blockfirst $1 W$2 N=$3 wall=$(( $(date +%s) - t0 ))s rc=$?" >> "$LOG"
done
echo "=== done blockfirst $(date -Is)" >> "$LOG"
