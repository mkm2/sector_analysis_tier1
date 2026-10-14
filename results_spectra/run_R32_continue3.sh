#!/bin/bash
# R32 continuation, part 3 (2026-10-14 18:30, user lifted the 18:35 cutoff; finish within 12 h).
# Runs every unit part 2 skipped at its deadline, then the X gate and further H sizes.
# R32 continuation (2026-10-14, user go-ahead): v2 pipeline (U never formed, compressed
# entanglement), per-rule maximum N, pbc check, optional X gate.
# One job at a time; no new job is started after DEADLINE (compute budget: 8 h total,
# ~1.7 h used before this script).  Finished units are skipped.
# Log: results_spectra/runlog_v2.txt
HERE="$(cd "$(dirname "$0")" && pwd)"; HSF="$(cd "$HERE/../../HSF" && pwd)"
LOG="$HERE/runlog_v2.txt"
DEADLINE=$(date -d "2026-10-15 04:30" +%s)     # restart after the 20:08 shutdown; leaves 2 h for analysis and report
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

# wait until part 2 has exited and no job is running (memory: one job at a time)
while pgrep -f "run_R32_continue2.sh" > /dev/null || pgrep -f "julia.*generate_spectral_data" > /dev/null; do sleep 30; done
echo "=== start part 3 $(date -Is)" >> "$LOG"
phi_ok() { ~/pyenvs/main/bin/python -c "import h5py,sys; sys.exit(0 if 'phi' in h5py.File(sys.argv[1],'r') else 1)" "$1"; }
# remainder of part 2 -- P3: obc0 extensions
for r in 12 10 6 9; do run H obc0 15 $r; done
for N in 15 16; do run H obc0 $N 0; done
for N in 15 16 17 18 19 20; do for r in 4 2 8; do run H obc0 $N $r; done; done
for N in 15 16 17 18 19; do run H obc0 $N 1; done
# P4: obc0 even-N reruns of the Omega rules (phi output)
for N in 6 8 10 12 14; do for r in 1 6 7 8 9 14; do
  f=$(ls "$HERE"/H/spec_N${N}_H_HSF${r}_W*.jld2 2>/dev/null)
  if [ -n "$f" ] && ! phi_ok "$f"; then rm -f "$f"; fi
  run H obc0 $N $r
done; done
# P5: remaining pbc N = 14
for r in 6 9 1 2 4 8 0 15; do run H pbc 14 $r; done
# P6: X gate (permutation circuits), obc0 N = 6..14 and rings N = 6..14
for N in 6 7 8 9 10 11 12 13 14; do for r in $(seq 0 15); do run X obc0 $N $r; done; done
for N in 6 8 10 12 14; do for r in $(seq 0 15); do run X pbc $N $r; done; done
# P7: further H sizes where the blocks allow: one-sided 153/195 at N = 15 (block 2^14),
# 60/102 at N = 16, XOR 150/105 up to N = 18
for r in 5 3; do run H obc0 15 $r; done
for r in 12 10; do run H obc0 16 $r; done
for N in 16 17 18; do for r in 6 9; do run H obc0 $N $r; done; done
echo "=== done part 3 $(date -Is)" >> "$LOG"
