#!/bin/bash
# Overnight series of H3 runs: light to heavy, up to the first out-of-memory failure.
#
#   nohup caffeinate -dimsu bash night_queue.sh > ~/models/logs/h3-night.log 2>&1 & disown
#
# Each run: per-phase timing (run_bench.py writes JSON) plus a memory profile
# (memwatch.sh writes CSV). A failing configuration is flagged and the series keeps
# going — a failure on the heaviest setting must not discard the lighter results.
#
# caffeinate, run outside this script, keeps the machine from sleeping for the
# duration of the series.

# Same reason as memwatch.sh: the awk `printf`s below format the numbers that end up in the
# summary, and under a comma-decimal locale they would emit "11,54" into a line the reader (and
# the next `awk`) parses as a number.
export LC_ALL=C

set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$PROJ/.venv/bin/python"
LOGS="$HOME/models/logs"
SUMMARY="$LOGS/h3-night-summary.txt"

cd "$PROJ" || exit 1
mkdir -p "$LOGS"

# tag:width:height:duration — in increasing order of load.
# 1344x768 is H3's native resolution; everything else is off-distribution.
#
# Every height and width here must be a multiple of 32 or the port refuses to pack the canvas.
# `half` shipped as 768x432 once and failed on exactly that, wasting its slot in the series:
# 432 = 13.5 * 32. 448 is the nearest legal height.
#
# `native15` (1344x768, 15 s) is deliberately absent. It was never attempted: the 10-second run
# measured 1881 s per step, and 15 s extrapolates past 35 hours for a single clip. Adding it back
# means committing a day and a half of machine time to a number the 5 s and 10 s runs already
# bracket — do that on purpose, not by leaving it in a queue that runs overnight.
RUNS="
smoke:512:512:2.4
half:768:448:2.4
native5:1344:768:5.0
native10:1344:768:10.0
"

{
  echo "=== H3 overnight series ==="
  echo "start: $(date '+%Y-%m-%d %H:%M')"
  echo "machine: M4 Pro, 48 GB, GPU limit $(sysctl -n iogpu.wired_limit_mb 2>/dev/null) MB"
  echo
} | tee -a "$SUMMARY"

while IFS=: read -r tag w h dur; do
  [ -z "$tag" ] && continue

  echo "--- $tag: ${w}x${h}, ${dur}s -- start $(date '+%H:%M') ---" | tee -a "$SUMMARY"
  started=$(date +%s)

  "$PY" run_bench.py --width "$w" --height "$h" --duration "$dur" \
      --steps 31 --tag "$tag" > "$LOGS/h3-gen-$tag.log" 2>&1 &
  gen_pid=$!

  # Attach the monitor immediately, to `$!` directly. This used to `sleep 10` and then `pgrep` for
  # the command line — which cost the most interesting ten seconds of the whole trace: the text
  # encoder loads in 8.3 s and is unloaded right after, so the 28 GB encoding phase was over before
  # the first sample was ever taken, and every published "peak RSS" described the diffusion phase
  # only. `$!` is already the python PID (bash execs a simple redirected command directly, without
  # an intervening subshell), so the pgrep was never needed and the sleep bought nothing.
  bash memwatch.sh "$gen_pid" "$LOGS/h3-mem-$tag.csv" 10 > "$LOGS/h3-mem-$tag.err" 2>&1 &

  wait "$gen_pid"
  rc=$?
  elapsed=$(( $(date +%s) - started ))

  if [ "$rc" -eq 0 ]; then
    sps=$(grep -oE "[0-9.]+s per step" "$LOGS/h3-gen-$tag.log" | tail -1)
    # Columns 2 and 5 of memwatch.sh's CSV: rss_gb and swap_used_gb. This is only true while that
    # CSV really has six fields — see the LC_ALL note at the top of memwatch.sh for the locale bug
    # that made it eleven, at which point $2 and $5 were the integer part of the RSS and the
    # *decimal* part of the wired figure, and this line reported a confident, wrong number.
    peak=$(LC_ALL=C awk -F, 'NR>1 {if($2>r)r=$2} END {printf "%.2f", r}' "$LOGS/h3-mem-$tag.csv" 2>/dev/null)
    swap=$(LC_ALL=C awk -F, 'NR>1 {if($5>s)s=$5} END {printf "%.2f", s}' "$LOGS/h3-mem-$tag.csv" 2>/dev/null)
    # `swap` is the machine's total swap in use, not this process's — `sysctl vm.swapusage` has no
    # per-process view. It is useful as a direction (does a run push the machine into swap?), never
    # as an attribution.
    echo "    DONE in $((elapsed / 60)) min | $sps | peak RSS ${peak} GB | system swap in use ${swap} GB" | tee -a "$SUMMARY"
  else
    reason=$(grep -iE "error|out of memory|Traceback" "$LOGS/h3-gen-$tag.log" | tail -1 | cut -c1-120)
    echo "    FAILED (code $rc) after $((elapsed / 60)) min: $reason" | tee -a "$SUMMARY"
  fi
done <<< "$RUNS"

{
  echo
  echo "=== series finished: $(date '+%Y-%m-%d %H:%M') ==="
  ls -la "$HOME/models/video-out"/*.mp4 2>/dev/null | awk '{printf "  %6.1f MB  %s\n", $5/1048576, $9}'
} | tee -a "$SUMMARY"
