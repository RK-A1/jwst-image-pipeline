#!/bin/bash
# run_overnight.sh — run the labelling pass unattended, safely.
#
# Guards against the two things that actually kill long local runs:
#   1. the machine idle-sleeping        -> caffeinate
#   2. the battery running out          -> refuse to start off AC, abort if it drains
#
# 01_label.py appends to JSONL and skips photo_ids already present, so aborting is
# free — rerun this script and it picks up where it stopped.
#
# Usage:
#   build/run_overnight.sh                          # text only
#   build/run_overnight.sh --vision                 # send images too
#   build/run_overnight.sh --model gemma4:e2b       # any 01_label.py flags pass through

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

MIN_BATTERY=25          # abort below this, so the machine never dies mid-write
CHECK_EVERY=60

on_ac()  { pmset -g batt | grep -q "AC Power"; }
charge() { pmset -g batt | grep -o '[0-9]\+%' | head -1 | tr -d '%'; }

# AC is preferred but not required — a short run on a full battery is fine. What is
# never fine is starting with little charge: macOS force-sleeps at critical battery
# regardless of caffeinate, and no assertion can override that.
START_BATTERY=50
if on_ac; then
    echo "on AC, battery $(charge)% — starting"
elif [ "$(charge)" -ge "$START_BATTERY" ]; then
    echo "on battery at $(charge)% — starting anyway; will stop below ${MIN_BATTERY}%."
    echo "Plug in if this is a long run."
else
    echo "REFUSING TO START: on battery at $(charge)%, below the ${START_BATTERY}% floor."
    echo "caffeinate cannot stop a critical-battery sleep. Plug in and retry."
    exit 1
fi

# Hold off idle sleep and disk sleep for as long as the labeller runs.
caffeinate -i -m python -u build/01_label.py "$@" &
JOB=$!
echo "labeller pid $JOB (wrapped in caffeinate)"

# Watchdog: if power is pulled or the battery drains, stop cleanly rather than
# letting macOS force-sleep the machine mid-request.
while kill -0 "$JOB" 2>/dev/null; do
    sleep "$CHECK_EVERY"
    if ! on_ac; then
        pct=$(charge)
        if [ "${pct:-0}" -lt "$MIN_BATTERY" ]; then
            echo ""
            echo "POWER LOST and battery at ${pct}% — stopping cleanly."
            echo "Progress is saved. Plug in and rerun this script to resume."
            kill "$JOB" 2>/dev/null
            wait "$JOB" 2>/dev/null
            exit 2
        fi
        echo "  warning: running on battery (${pct}%) — will stop below ${MIN_BATTERY}%"
    fi
done

wait "$JOB"
status=$?
echo "labeller exited with status $status"
[ "$status" -eq 0 ] && echo "done — next: python build/02_assemble.py"
exit "$status"
