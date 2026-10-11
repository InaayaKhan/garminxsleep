"""Where the dashboard reads its data and how that data is interpreted.

By default everything comes from the synthetic dataset in ``sample_data/``.
To use your own data, point ``GARMINXSLEEP_DATA_DIR`` at a folder with the
same layout (keep it outside Git, for example ``private/``):

    <data dir>/
        settings.json            timezone and Garmin start date (optional)
        sleep_history_*.csv      Sleepmeter export(s); the newest is used
        aid_colors.json          per-aid chart colours (optional)
        garmin_*_overrides.json  manual merge corrections (optional)
        raw/<collector>/*.json   Garmin responses written by garmin_sync
"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(
    os.getenv("GARMINXSLEEP_DATA_DIR") or PROJECT_ROOT / "sample_data"
).expanduser().resolve()
RAW_ROOT = DATA_ROOT / "raw"
DATABASE_PATH = PROJECT_ROOT / "data" / "sleep_dashboard.duckdb"

_SETTINGS_PATH = DATA_ROOT / "settings.json"
_SETTINGS: dict[str, str] = (
    json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    if _SETTINGS_PATH.exists()
    else {}
)

# IANA name used to show Garmin's UTC timestamps as local time.
TIMEZONE_NAME = _SETTINGS.get("timezone", "UTC")


def garmin_start_date() -> date:
    """First day Garmin is authoritative; earlier days come from Sleepmeter.

    Taken from ``settings.json``; otherwise the first day with Garmin sleep
    data, so Sleepmeter covers everything before the watch was worn.
    """
    configured = _SETTINGS.get("garmin_start_date")
    if configured:
        return date.fromisoformat(configured)
    sleep_dates = []
    for path in (RAW_ROOT / "sleep").glob("*.json"):
        try:
            sleep_dates.append(date.fromisoformat(path.stem))
        except ValueError:
            continue
    return min(sleep_dates, default=date.max)
