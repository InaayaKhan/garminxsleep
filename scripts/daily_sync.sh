#!/bin/zsh
# Run the Garmin sync once, unattended, appending output to <data dir>/sync.log.
#
# Catches up on days missed while the computer was off (--catch-up), and does
# nothing if a complete sync already ran in the last 20 hours, so it's safe to
# run both on a schedule and at every login.
#
# Uses the saved Garmin token. It never waits for keyboard input: if the token
# has expired, the run fails and the log says so; then run
# `python -m dashboard.garmin_sync` yourself once to log in again.
#
#   GARMINXSLEEP_DATA_DIR  data folder (default: <project>/private)
#   GARMINXSLEEP_PYTHON    Python with the project installed
#                          (default: <project>/.venv/bin/python)
set -euo pipefail

project=${0:A:h:h}
python=${GARMINXSLEEP_PYTHON:-$project/.venv/bin/python}
export GARMINXSLEEP_DATA_DIR=${GARMINXSLEEP_DATA_DIR:-$project/private}
log=$GARMINXSLEEP_DATA_DIR/sync.log

mkdir -p "$GARMINXSLEEP_DATA_DIR"
cd "$project"
{
  print "=== $(date '+%Y-%m-%d %H:%M:%S')"
  "$python" -m dashboard.garmin_sync --catch-up --skip-if-synced-within 20 </dev/null
} >>"$log" 2>&1
