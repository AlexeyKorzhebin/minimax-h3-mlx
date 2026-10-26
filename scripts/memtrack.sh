#!/bin/bash
# Records peak memory per run, with the instrument that can actually see it.
#
# `ps` RSS misses the Metal allocations entirely -- measured against this same process, 9.3 GB
# reported against 25 GB real. `footprint -p` reports `phys_footprint` and, for free,
# `phys_footprint_peak`, which is the number a long run has to be judged on: the peak decides
# whether the machine swaps, and swapping showed up once as a step taking 818 s where its
# neighbours took 568.
set -u
export LC_ALL=C          # awk's printf otherwise writes a decimal comma under a ru locale
DEST=${1:-$HOME/h3-outputs/memory.tsv}
[ -f "$DEST" ] || printf 'time\ttag\tphase\tfootprint_gb\tpeak_gb\tfree_gb\tswap_gb\n' > "$DEST"

while true; do
  PID=$(pgrep -f "m h3_48gb generate" | head -1)
  if [ -n "$PID" ]; then
    FP=$(/usr/bin/footprint -p "$PID" 2>/dev/null | grep -E 'phys_footprint(_peak)?:' \
         | awk '{print $2, $3}' | tr '\n' ' ')
    # footprint prints "25 GB" or "812 MB"; normalise both to GB
    read -r a1 u1 a2 u2 <<< "$FP"
    to_gb () { case "$2" in MB) echo "scale=2; $1/1024" | bc;; KB) echo 0;; *) echo "$1";; esac; }
    CUR=$(to_gb "${a1:-0}" "${u1:-GB}")
    PEAK=$(to_gb "${a2:-0}" "${u2:-GB}")
    # The log this process is actually writing to, via its own stdout descriptor. Guessing by
    # mtime picked a finished run's log instead, and a peak attributed to the wrong phase is
    # worse than no attribution.
    LOG=$(lsof -p "$PID" -a -d 1 -Fn 2>/dev/null | sed -n 's/^n//p' | head -1)
    TAG=$(basename "${LOG:-unknown}" .log)
    PHASE=$(grep -oE "step [0-9]+/[0-9]+|loaded [a-z ]+|unloaded [a-z ]+" "${LOG:-/dev/null}" 2>/dev/null | tail -1)
    FREE=$(vm_stat | awk '/Pages free/ {f=$3} /Pages inactive/ {i=$3} END {gsub(/\./,"",f); gsub(/\./,"",i); printf "%.2f", (f+i)*16384/1073741824}')
    SWAP=$(sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); printf "%.2f", $6/1024}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%H:%M)" "$TAG" "${PHASE:-?}" \
      "$CUR" "$PEAK" "$FREE" "$SWAP" >> "$DEST"
  fi
  sleep 60
done
