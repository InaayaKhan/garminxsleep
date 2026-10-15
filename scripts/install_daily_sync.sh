#!/bin/zsh
# Schedule scripts/daily_sync.sh to run every day at a set time (macOS launchd).
#
#   scripts/install_daily_sync.sh 07:30      install, or change the time
#   scripts/install_daily_sync.sh --uninstall
#
# The job runs as you while you're logged in: at the set time, when the Mac
# wakes if it was asleep then, and at every login (so a Mac that was off at
# that time catches up). daily_sync.sh skips runs within 20 hours of the last
# complete sync and fetches any missed days. Output: <data dir>/sync.log.
set -euo pipefail

label=com.garminxsleep.daily-sync
plist=$HOME/Library/LaunchAgents/$label.plist
project=${0:A:h:h}

if [[ ${1:-} == --uninstall ]]; then
  launchctl bootout "gui/$UID/$label" 2>/dev/null || true
  rm -f "$plist"
  print "Removed the daily sync."
  exit 0
fi

if [[ ! ${1:-} =~ '^([01]?[0-9]|2[0-3]):[0-5][0-9]$' ]]; then
  print -u2 "Usage: $0 HH:MM   (24-hour time, e.g. 07:30)   or   $0 --uninstall"
  exit 1
fi
hour=$((10#${1%%:*}))
minute=$((10#${1##*:}))

python=${GARMINXSLEEP_PYTHON:-$project/.venv/bin/python}
data_dir=${GARMINXSLEEP_DATA_DIR:-$project/private}
if [[ ! -x $python ]]; then
  print -u2 "No Python found at $python. Create the venv first (see README) or set GARMINXSLEEP_PYTHON."
  exit 1
fi

mkdir -p "${plist:h}"
cat >"$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>$project/scripts/daily_sync.sh</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>GARMINXSLEEP_PYTHON</key><string>$python</string>
    <key>GARMINXSLEEP_DATA_DIR</key><string>$data_dir</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>$hour</integer>
    <key>Minute</key><integer>$minute</integer>
  </dict>
</dict>
</plist>
EOF

launchctl bootout "gui/$UID/$label" 2>/dev/null || true
launchctl bootstrap "gui/$UID" "$plist"
printf "Daily sync scheduled at %02d:%02d. Log: %s/sync.log\n" "$hour" "$minute" "$data_dir"
