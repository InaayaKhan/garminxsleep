"""Records: the raw session table."""

from __future__ import annotations

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
    st.dataframe(display, width="stretch", hide_index=True)
