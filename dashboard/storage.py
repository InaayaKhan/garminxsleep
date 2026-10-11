from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from dashboard.model import SleepDataset, split_intervals_by_calendar_day


def _database_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == "object" and not result.empty:
            sample = next((value for value in result[column] if value is not None), None)
            if hasattr(sample, "isoformat"):
                result[column] = result[column].map(
                    lambda value: value.isoformat() if value is not None else None
                )
    return result


def persist_dataset(dataset: SleepDataset, database_path: str | Path) -> Path:
    path = Path(database_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = {
        "sleep_aids": dataset.aids,
        "sleep_sessions": dataset.sessions,
        "sleep_intervals": dataset.intervals,
        "sleep_stages": dataset.sleep_stages,
        "garmin_activities": dataset.activities,
        "garmin_menstrual": dataset.menstrual,
        "sleep_chart_segments": split_intervals_by_calendar_day(dataset.intervals),
        "daily_sleep": dataset.daily,
        "garmin_daily_metrics": dataset.garmin_metrics,
    }

    with duckdb.connect(str(path)) as connection:
        for table_name, frame in frames.items():
            normalized = _database_frame(frame)
            connection.register("incoming_frame", normalized)
            connection.execute(
                f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM incoming_frame'
            )
            connection.unregister("incoming_frame")
        connection.execute(
            """
            CREATE OR REPLACE TABLE dashboard_metadata AS
            SELECT
                ?::VARCHAR AS source_file,
                ?::DATE AS coverage_start,
                ?::DATE AS coverage_end,
                current_timestamp AS rebuilt_at
            """,
            [
                dataset.source_path.name,
                dataset.coverage_start.isoformat(),
                dataset.coverage_end.isoformat(),
            ],
        )
    return path
