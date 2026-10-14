#!/bin/bash
# Regenerate spectra written before s2_block_rand was added (2026-10-14 ~02:05).
# Waits for run_H.sh to finish, then reruns every file lacking the field.
HERE="$(cd "$(dirname "$0")" && pwd)"; HSF="$(cd "$HERE/../../HSF" && pwd)"; LOG="$HERE/runlog_H.txt"
while pgrep -f run_H.sh > /dev/null; do sleep 60; done
echo "=== fixup start $(date -Is)" >> "$LOG"
for f in $(~/pyenvs/main/bin/python -c "
import h5py, glob
for f in sorted(glob.glob('$HERE/H/*.jld2')):
    if 's2_block_rand' not in h5py.File(f,'r'): print(f)"); do
  b=$(basename "$f"); N=$(echo "$b" | sed -E 's/spec_N([0-9]+)_.*/\1/'); r=$(echo "$b" | sed -E 's/.*_HSF([0-9]+)_.*/\1/')
  t0=$(date +%s); rm -f "$f"
  (cd "$HSF" && julia --project=. -t 20 generate_spectral_data.jl "$N" "$r" H "$HERE/H") >> "$LOG" 2>&1
  echo "$(date -Is) fixup HSF$r N=$N wall=$(( $(date +%s) - t0 ))s rc=$?" >> "$LOG"
done
echo "=== fixup done $(date -Is)" >> "$LOG"
