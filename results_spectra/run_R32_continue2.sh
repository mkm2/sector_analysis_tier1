#!/bin/bash
# R32 continuation, part 2 (after the pbc symmetry fix: Omega_T = T1 L_A, phi output)
# R32 continuation (2026-10-14, user go-ahead): v2 pipeline (U never formed, compressed
# entanglement), per-rule maximum N, pbc check, optional X gate.
# One job at a time; no new job is started after DEADLINE (compute budget: 8 h total,
# ~1.7 h used before this script).  Finished units are skipped.
# Log: results_spectra/runlog_v2.txt
HERE="$(cd "$(dirname "$0")" && pwd)"; HSF="$(cd "$HERE/../../HSF" && pwd)"
LOG="$HERE/runlog_v2.txt"
DEADLINE=$(date -d "2026-10-14 18:35" +%s)     # same absolute deadline as run_R32_continue.sh
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

echo "=== start part 2 $(date -Is)" >> "$LOG"
# never overlap with a running job (memory)
while pgrep -f "julia.*generate_spectral_data" > /dev/null; do sleep 20; done
# P1: pbc N = 6..12 regenerated with Omega_T (old T2-only files removed beforehand)
for N in 6 8 10 12; do for r in $(seq 0 15); do run H pbc $N $r; done; done
# P2: pbc N = 14, chaotic (k=3) and one-sided first
for r in 13 11 7 14 12 10 5 3; do run H pbc 14 $r; done
# P3: remaining obc0 extensions
for r in 12 10 6 9; do run H obc0 15 $r; done
for N in 15 16; do run H obc0 $N 0; done
for N in 15 16 17 18 19 20; do for r in 4 2 8; do run H obc0 $N $r; done; done
for N in 15 16 17 18 19; do run H obc0 $N 1; done
# P4: obc0 even-N reruns of the Omega rules (phi output), small N first
for N in 6 8 10 12; do for r in 1 6 7 8 9 14; do
  f=$(ls "$HERE"/H/spec_N${N}_H_HSF${r}_W*.jld2 2>/dev/null)
  if [ -n "$f" ] && ! ~/pyenvs/main/bin/python -c "import h5py,sys; sys.exit(0 if 'phi' in h5py.File(sys.argv[1],'r') else 1)" "$f"; then rm -f "$f"; fi
  run H obc0 $N $r
done; done
for r in 7 14; do
  f=$(ls "$HERE"/H/spec_N14_H_HSF${r}_W*.jld2 2>/dev/null)
  if [ -n "$f" ] && ! ~/pyenvs/main/bin/python -c "import h5py,sys; sys.exit(0 if 'phi' in h5py.File(sys.argv[1],'r') else 1)" "$f"; then rm -f "$f"; fi
  run H obc0 14 $r
done
# P5: remaining pbc N = 14, then the optional X gate
for r in 6 9 1 2 4 8 0 15; do run H pbc 14 $r; done
for N in 6 7 8 9 10 11 12; do for r in $(seq 0 15); do run X obc0 $N $r; done; done
echo "=== done part 2 $(date -Is)" >> "$LOG"
