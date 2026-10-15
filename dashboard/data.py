"""Locating, loading, and caching the dataset, and saving manual aid overrides."""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import streamlit as st

from dashboard.config import DATA_ROOT, RAW_ROOT, TIMEZONE_NAME, garmin_start_date
from dashboard.model import SleepDataset, load_combined_sleep_data

GARMIN_RAW_ROOTS = [RAW_ROOT]
GARMIN_AID_OVERRIDES_PATH = DATA_ROOT / "garmin_aid_overrides.json"
GARMIN_SCORE_OVERRIDES_PATH = DATA_ROOT / "garmin_score_overrides.json"
GARMIN_MERGE_OVERRIDES_PATH = DATA_ROOT / "garmin_merge_overrides.json"


def find_export() -> Path:
    configured = os.getenv("SLEEPMETER_EXPORT")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Configured export does not exist: {path}")
        return path
    exports = sorted(
        DATA_ROOT.glob("sleep_history_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not exports:
        raise FileNotFoundError(f"No sleep_history_*.csv export found in {DATA_ROOT}")
    return exports[0]


def garmin_archive_signature() -> tuple[tuple[str, int, int], ...]:
    paths = [
        path
        for root in GARMIN_RAW_ROOTS
        if root.exists()
        for path in root.glob("*/*.json")
    ]
    if GARMIN_AID_OVERRIDES_PATH.exists():
        paths.append(GARMIN_AID_OVERRIDES_PATH)
    if GARMIN_SCORE_OVERRIDES_PATH.exists():
        paths.append(GARMIN_SCORE_OVERRIDES_PATH)
    if GARMIN_MERGE_OVERRIDES_PATH.exists():
        paths.append(GARMIN_MERGE_OVERRIDES_PATH)
    return tuple(
        sorted((str(path), path.stat().st_mtime_ns, path.stat().st_size) for path in paths)
    )


def save_garmin_aid_override(sleep_date: date, aid_name: str | None) -> None:
    overrides: dict[str, str] = {}
    if GARMIN_AID_OVERRIDES_PATH.exists():
        overrides = json.loads(GARMIN_AID_OVERRIDES_PATH.read_text(encoding="utf-8"))
    key = sleep_date.isoformat()
    overrides[key] = aid_name or "None"
    GARMIN_AID_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = GARMIN_AID_OVERRIDES_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(overrides, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(GARMIN_AID_OVERRIDES_PATH)


@st.cache_data(show_spinner=False)
def cached_dataset(
    path_text: str,
    modified_ns: int,
    raw_signature: tuple[tuple[str, int, int], ...],
) -> SleepDataset:
    del modified_ns, raw_signature
    return load_combined_sleep_data(
        path_text,
        GARMIN_RAW_ROOTS,
        garmin_start_date=garmin_start_date(),
        aid_overrides_path=GARMIN_AID_OVERRIDES_PATH,
        score_overrides_path=GARMIN_SCORE_OVERRIDES_PATH,
        merge_overrides_path=GARMIN_MERGE_OVERRIDES_PATH,
        timezone_name=TIMEZONE_NAME,
    )
