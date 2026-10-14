#!/bin/bash
# R32 production runs, gate H.  One job at a time (dense LAPACK is fastest that way).
# Usage: bash results_spectra/run_H.sh  (from the repo root); log -> results_spectra/runlog_H.txt
# Each line of the log: date, rule (HSF / Wolfram), N, wall seconds, exit code.
HERE="$(cd "$(dirname "$0")" && pwd)"
HSF="$(cd "$HERE/../../HSF" && pwd)"
OUT="$HERE/H"
LOG="$HERE/runlog_H.txt"
mkdir -p "$OUT"
run() {   # N rule
  local f
  f=$(ls "$OUT"/spec_N$1_H_HSF$2_W*.jld2 2>/dev/null)
  if [ -n "$f" ]; then echo "skip N=$1 HSF$2 (exists)" >> "$LOG"; return; fi
  local t0=$(date +%s)
  (cd "$HSF" && julia --project=. -t 20 generate_spectral_data.jl "$1" "$2" H "$OUT") >> "$LOG" 2>&1
  local rc=$?
  echo "$(date -Is) HSF$2 N=$1 wall=$(( $(date +%s) - t0 ))s rc=$rc" >> "$LOG"
}
echo "=== start $(date -Is)" >> "$LOG"
# stage 1: all 16 rules, N = 6..13
for N in 6 7 8 9 10 11 12 13; do for r in $(seq 0 15); do run $N $r; done; done
# stage 2: all 16 rules, N = 14
for r in 1 2 4 8 6 9 10 12 0 15 14 7 3 5 11 13; do run 14 $r; done
# stage 3: polynomial rules (largest sector <= 16384), N = 15
for r in 6 9 10 12; do run 15 $r; done
# stage 4: exponentially fragmented rules, N = 15..18
for N in 15 16 17 18; do for r in 1 2 4 8; do run $N $r; done; done
echo "=== done $(date -Is)" >> "$LOG"
