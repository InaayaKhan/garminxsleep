"""Dashboard tabs. Each module renders one tab from a shared ``ViewData``."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from dashboard.model import SleepDataset


@dataclass(frozen=True)
class Frames:
    """Every dataset table limited to one date range."""

    daily: pd.DataFrame
    sessions: pd.DataFrame
    intervals: pd.DataFrame
    night_intervals: pd.DataFrame
    sleep_stages: pd.DataFrame
    activities: pd.DataFrame
    menstrual: pd.DataFrame
    garmin_metrics: pd.DataFrame
    intraday: pd.DataFrame


def frames_between(dataset: SleepDataset, start: date | None, end: date) -> Frames:
    """Limit every table to ``start``..``end``; ``start=None`` keeps all history."""

    def by_date(frame: pd.DataFrame, column: str) -> pd.DataFrame:
        in_range = frame[column] <= end
        if start is not None:
            in_range &= frame[column] >= start
        return frame[in_range].copy()

    sessions = by_date(dataset.sessions, "sleep_date")
    intervals = dataset.intervals[
        dataset.intervals["session_id"].isin(sessions["session_id"])
    ].copy()
    night_ids = sessions.loc[sessions["session_type"] == "NIGHT_SLEEP", "session_id"]
    return Frames(
        daily=by_date(dataset.daily, "date"),
        sessions=sessions,
        intervals=intervals,
        night_intervals=intervals[intervals["session_id"].isin(night_ids)].copy(),
        sleep_stages=dataset.sleep_stages[
            dataset.sleep_stages["session_id"].isin(sessions["session_id"])
        ].copy(),
        activities=by_date(dataset.activities, "activity_date"),
        menstrual=by_date(dataset.menstrual, "date"),
        garmin_metrics=by_date(dataset.garmin_metrics, "date"),
        intraday=by_date(dataset.intraday, "date"),
    )


@dataclass(frozen=True)
class ViewData:
    """What every tab renders from.

    ``selected`` covers the sidebar date range. ``history`` covers everything
    up to its end date, for charts that open on the range but pan further back.
    """

    dataset: SleepDataset
    start_date: date
    end_date: date
    selected: Frames
    history: Frames
