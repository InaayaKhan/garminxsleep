"""Records: the raw session table."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.tabs import ViewData


def render(view: ViewData) -> None:
    st.markdown(
        'Every sleep session in the selected date range.'
    )
    display = view.selected.sessions[
        [
            "sleep_date",
            "sleep_start",
            "wake_time",
            "session_type",
            "sleep_minutes",
            "interruption_minutes",
            "aid_name",
            "legacy_garmin_score_10",
            "zero_duration",
            "holes",
            "dreams",
            "hindrances",
            "notes",
        ]
    ].copy()
    display["sleep_hours"] = (display.pop("sleep_minutes") / 60).round(2)
    # Local clock times (offsets dropped) so the columns sort and format as dates.
    for column in ("sleep_start", "wake_time"):
        display[column] = pd.to_datetime(
            display[column].map(lambda value: value.replace(tzinfo=None))
        )
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "sleep_date": st.column_config.DateColumn("sleep_date", format="DD.MM.YYYY"),
            "sleep_start": st.column_config.DatetimeColumn("sleep_start", format="DD.MM.YYYY HH:mm"),
            "wake_time": st.column_config.DatetimeColumn("wake_time", format="DD.MM.YYYY HH:mm"),
        },
    )
