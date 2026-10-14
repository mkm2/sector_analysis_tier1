#!/bin/bash
# R32 production runs, gate X.  One job at a time (dense LAPACK is fastest that way).
# Usage: bash results_spectra/run_X.sh  (from the repo root); log -> results_spectra/runlog_X.txt
# Each line of the log: date, rule (HSF / Wolfram), N, wall seconds, exit code.
HERE="$(cd "$(dirname "$0")" && pwd)"
HSF="$(cd "$HERE/../../HSF" && pwd)"
OUT="$HERE/X"
LOG="$HERE/runlog_X.txt"
mkdir -p "$OUT"
while pgrep -f "run_H.sh|fixup_H.sh|run_H_pbc.sh" > /dev/null; do sleep 60; done
run() {   # N rule
  local f
  f=$(ls "$OUT"/spec_N$1_X_HSF$2_W*.jld2 2>/dev/null)
  if [ -n "$f" ]; then echo "skip N=$1 HSF$2 (exists)" >> "$LOG"; return; fi
  local t0=$(date +%s)
  (cd "$HSF" && julia --project=. -t 20 generate_spectral_data_v2.jl "$1" "$2" X "$OUT" obc0) >> "$LOG" 2>&1
  local rc=$?
  echo "$(date -Is) HSF$2 N=$1 wall=$(( $(date +%s) - t0 ))s rc=$rc" >> "$LOG"
}
echo "=== start $(date -Is)" >> "$LOG"
# X gate (optional part of R32): all 16 rules, N = 6..12; then N = 13 (budget permitting)
for N in 6 7 8 9 10 11 12; do for r in $(seq 0 15); do run $N $r; done; done
for r in $(seq 0 15); do run 13 $r; done
echo "=== done $(date -Is)" >> "$LOG"
