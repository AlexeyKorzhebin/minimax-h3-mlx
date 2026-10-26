#!/bin/bash
# Background memory monitor for an H3 generation run.
#
#   bash memwatch.sh <pid> <csv> [interval_seconds]
#
# Writes a CSV: elapsed time, process RSS, GPU allocator (wired) memory, compressed
# memory, swap. Needed to tell "the model fit" apart from "the model fit at the cost
# of swap" — the second looks the same at a glance, just several times slower.

# LC_ALL=C is load-bearing, not hygiene. `printf` in bash and in awk both honour
# LC_NUMERIC, so under a comma-decimal locale (ru_RU, de_DE, fr_FR, ...) `%.2f`
# emits "11,54" and every value lands as two comma-separated fields — a 6-column CSV
# silently becomes an 11-column one. Worse, bash's own printf then *rejects* bc's
# dot-decimal output as a malformed number and writes 0,00 for wired and compressed.
# This corrupted every sample of the overnight series that produced docs/RESULTS.md;
# those CSVs had to be re-parsed by hand. Pin the numeric locale before the first
# sample, not after.
export LC_ALL=C

PID="${1:?pass a PID}"
OUT="${2:?pass a path for the CSV}"
INTERVAL="${3:-5}"

echo "elapsed_s,rss_gb,wired_gb,compressed_gb,swap_used_gb,free_gb" > "$OUT"
START=$(date +%s)

while kill -0 "$PID" 2>/dev/null; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - START))

  RSS=$(ps -o rss= -p "$PID" 2>/dev/null | awk '{printf "%.2f", $1/1048576}')
  [ -z "$RSS" ] && break

  eval "$(vm_stat | awk '
    /Pages free/          {gsub(/\./,"",$3); printf "F=%s;", $3}
    /Pages wired down/    {gsub(/\./,"",$4); printf "W=%s;", $4}
    /occupied by compressor/ {gsub(/\./,"",$5); printf "C=%s;", $5}
  ')"
  SWAP=$(sysctl -n vm.swapusage 2>/dev/null | awk '{gsub(/M/,"",$6); printf "%.2f", $6/1024}')

  printf '%s,%s,%.2f,%.2f,%s,%.2f\n' \
    "$ELAPSED" "$RSS" \
    "$(echo "$W * 16384 / 1073741824" | bc -l)" \
    "$(echo "$C * 16384 / 1073741824" | bc -l)" \
    "$SWAP" \
    "$(echo "$F * 16384 / 1073741824" | bc -l)" >> "$OUT"

  sleep "$INTERVAL"
done

echo "=== peaks for this run ===" >&2
awk -F, 'NR>1 {if ($2>r) r=$2; if ($3>w) w=$3; if ($5>s) s=$5} END {
  printf "  RSS max:        %.2f GB\n  wired max:      %.2f GB\n  swap max:       %.2f GB\n", r, w, s}' "$OUT" >&2
