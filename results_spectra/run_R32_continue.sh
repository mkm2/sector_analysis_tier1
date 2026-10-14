#!/bin/bash
# R32 continuation (2026-10-14, user go-ahead): v2 pipeline (U never formed, compressed
# entanglement), per-rule maximum N, pbc check, optional X gate.
# One job at a time; no new job is started after DEADLINE (compute budget: 8 h total,
# ~1.7 h used before this script).  Finished units are skipped.
# Log: results_spectra/runlog_v2.txt
HERE="$(cd "$(dirname "$0")" && pwd)"; HSF="$(cd "$HERE/../../HSF" && pwd)"
LOG="$HERE/runlog_v2.txt"
DEADLINE=$(( $(date +%s) + 6 * 3600 ))
JL="julia --project=. -t 20 --heap-size-hint=14G generate_spectral_data_v2.jl"

run() {   # gate bc N rule
  local g=$1 bc=$2 N=$3 r=$4 dir
  dir="$HERE/$g"; [ "$bc" = pbc ] && dir="$HERE/${g}_pbc"
  mkdir -p "$dir"
  if ls "$dir"/spec_N${N}_${g}_HSF${r}_W*.jld2 >/dev/null 2>&1; then return; fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then echo "$(date -Is) DEADLINE: skip $g $bc N=$N HSF$r" >> "$LOG"; return; fi
  local t0=$(date +%s)
  (cd "$HSF" && $JL "$N" "$r" "$g" "$dir" "$bc") >> "$LOG" 2>&1
  local rc=$?
  echo "$(date -Is) $g $bc HSF$r N=$N wall=$(( $(date +%s) - t0 ))s rc=$rc" >> "$LOG"
}

echo "=== start $(date -Is)" >> "$LOG"
# S0: regenerate early obc0 files that lack s2_block_rand
# (read whole lines: the repository path contains spaces)
~/pyenvs/main/bin/python -c "
import h5py, glob
for f in sorted(glob.glob('$HERE/H/*.jld2')):
    if 's2_block_rand' not in h5py.File(f, 'r'): print(f)" > "$HERE/.s0_list"
while IFS= read -r f; do
  [ -z "$f" ] && continue
  b=$(basename "$f"); N=$(echo "$b" | sed -E 's/spec_N([0-9]+)_.*/\1/'); r=$(echo "$b" | sed -E 's/.*_HSF([0-9]+)_.*/\1/')
  rm -f "$f"; run H obc0 "$N" "$r"
done < "$HERE/.s0_list"
rm -f "$HERE/.s0_list"
# S1: obc0 N=14 for the single-sector rules (k=3 first)
for r in 13 7 14 11 5 3 15; do run H obc0 14 $r; done
# S2: pbc, all 16 rules, even N = 6..12
for N in 6 8 10 12; do for r in $(seq 0 15); do run H pbc $N $r; done; done
# S4: obc0 extensions to each rule's own maximum N
for r in 12 10 6 9; do run H obc0 15 $r; done                    # 60 102 150 105
for N in 15 16; do run H obc0 $N 0; done                         # 204
for N in 15 16 17 18 19 20; do for r in 4 2 8; do run H obc0 $N $r; done; done   # 156 198 108
for N in 15 16 17 18 19; do run H obc0 $N 1; done                # 201 (largest sector F_{N+2})
# S3: pbc N = 14 (k=3 and one-sided first)
for r in 13 11 7 14 12 10 5 3 6 9 1 2 4 8 0 15; do run H pbc 14 $r; done
# S5: optional X gate, obc0, N = 6..12
for N in 6 7 8 9 10 11 12; do for r in $(seq 0 15); do run X obc0 $N $r; done; done
echo "=== done $(date -Is)" >> "$LOG"
