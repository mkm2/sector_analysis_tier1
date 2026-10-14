#!/bin/bash
# R32 part 4: 201 at N = 20 so that the whole k = 1 class (108/156/198/201) reaches the same N
# (user request).  Expected peak ~20-23 GB (2.6 GB at N=18, 7.7 GB at N=19), so it runs LAST,
# alone, under a watchdog that kills only this job if available memory drops below 3 GB.
HERE="$(cd "$(dirname "$0")" && pwd)"; HSF="$(cd "$HERE/../../HSF" && pwd)"
LOG="$HERE/runlog_v2.txt"
DEADLINE=$(date -d "2026-10-15 04:30" +%s)     # restart after the 20:08 shutdown; leaves 2 h for analysis and report
while pgrep -f "run_R32_continue3.sh" > /dev/null || pgrep -f "julia.*generate_spectral_data" > /dev/null; do sleep 30; done
ls "$HERE"/H/spec_N20_H_HSF1_W*.jld2 >/dev/null 2>&1 && exit 0
if [ "$(date +%s)" -gt "$DEADLINE" ]; then echo "$(date -Is) DEADLINE: skip H obc0 N=20 HSF1" >> "$LOG"; exit 0; fi
echo "=== start part 4 (201 N=20, watchdog 3 GB) $(date -Is)" >> "$LOG"
t0=$(date +%s)
(cd "$HSF" && exec julia --project=. -t 20 --heap-size-hint=20G generate_spectral_data_v2.jl 20 1 H "$HERE/H" obc0) >> "$LOG" 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  avail=$(free -m | awk '/^Speicher|^Mem/{print $7}')
  if [ "$avail" -lt 3000 ]; then
    echo "$(date -Is) WATCHDOG: available ${avail} MB < 3000 MB, killing 201 N=20" >> "$LOG"
    pkill -f "julia.*generate_spectral_data_v2.jl 20 1 H"; break
  fi
  sleep 5
done
wait $pid; rc=$?
echo "$(date -Is) H obc0 HSF1 N=20 wall=$(( $(date +%s) - t0 ))s rc=$rc" >> "$LOG"
echo "=== done part 4 $(date -Is)" >> "$LOG"
