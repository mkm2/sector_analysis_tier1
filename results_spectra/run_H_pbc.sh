#!/bin/bash
# R32: H gate with periodic boundary conditions (even N only), v2 script.
# Waits for the obc0 production (run_H.sh) and its fixup to finish.
HERE="$(cd "$(dirname "$0")" && pwd)"; HSF="$(cd "$HERE/../../HSF" && pwd)"
OUT="$HERE/H_pbc"; LOG="$HERE/runlog_H_pbc.txt"; mkdir -p "$OUT"
while pgrep -f "run_H.sh|fixup_H.sh" > /dev/null; do sleep 60; done
run() {
  f=$(ls "$OUT"/spec_N$1_H_HSF$2_W*.jld2 2>/dev/null)
  if [ -n "$f" ]; then echo "skip N=$1 HSF$2" >> "$LOG"; return; fi
  t0=$(date +%s)
  (cd "$HSF" && julia --project=. -t 20 generate_spectral_data_v2.jl "$1" "$2" H "$OUT" pbc) >> "$LOG" 2>&1
  echo "$(date -Is) HSF$2 N=$1 pbc wall=$(( $(date +%s) - t0 ))s rc=$?" >> "$LOG"
}
echo "=== start $(date -Is)" >> "$LOG"
for N in 6 8 10 12; do for r in $(seq 0 15); do run $N $r; done; done
for r in 1 2 4 8 6 9 10 12 0 15 3 5 14 7 11 13; do run 14 $r; done
echo "=== done $(date -Is)" >> "$LOG"
