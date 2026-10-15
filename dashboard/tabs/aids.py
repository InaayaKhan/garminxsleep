"""Aids: per-aid comparisons, and editing the aid of Garmin nights."""

from __future__ import annotations

import streamlit as st

from dashboard.charts import (
    aid_duration_timeline_figure,
    aid_figure,
    aid_score_figure,
    aid_sleep_stage_composition_figure,
)
from dashboard.data import cached_dataset, save_garmin_aid_override
from dashboard.tabs import ViewData


def render(view: ViewData) -> None:
    garmin_aid_sessions = view.selected.sessions[
        (view.selected.sessions["source"] == "garmin")
        & (view.selected.sessions["session_type"] == "NIGHT_SLEEP")
    ].sort_values("sleep_date", ascending=False)
    with st.expander("Set aid for Garmin sleep"):
        if garmin_aid_sessions.empty:
            st.info("No Garmin night-sleep records are available in this date range.")
        else:
            session_dates = garmin_aid_sessions["sleep_date"].tolist()
            selected_aid_date = st.selectbox(
                "Garmin sleep date",
                session_dates,
                format_func=lambda value: value.isoformat(),
            )
            selected_aid_session = garmin_aid_sessions[
                garmin_aid_sessions["sleep_date"] == selected_aid_date
            ].iloc[0]
            current_aid = str(selected_aid_session["aid_name"])
            known_aids = sorted(
                {
                    "None",
                    *view.dataset.aids["aid_name"].astype(str).tolist(),
                    *view.dataset.sessions["aid_name"].dropna().astype(str).tolist(),
                },
                key=lambda value: (value != "None", value.casefold()),
            )
            aid_choices = known_aids + ["Custom…"]
            selected_aid = st.selectbox(
                "Recorded sleep aid",
                aid_choices,
                index=(
                    aid_choices.index(current_aid)
                    if current_aid in aid_choices
                    else aid_choices.index("Custom…")
                ),
                key=f"garmin_aid_choice_{selected_aid_date.isoformat()}",
            )
            custom_aid = ""
            if selected_aid == "Custom…":
                custom_aid = st.text_input(
                    "Custom aid name",
                    value=current_aid if current_aid not in known_aids else "",
                    key=f"garmin_custom_aid_{selected_aid_date.isoformat()}",
                ).strip()
            save_column, clear_column = st.columns(2)
            with save_column:
                if st.button("Save aid", type="primary", width="stretch"):
                    aid_to_save = custom_aid if selected_aid == "Custom…" else selected_aid
                    if not aid_to_save:
                        st.error("Enter a custom aid name before saving.")
                    else:
                        save_garmin_aid_override(selected_aid_date, aid_to_save)
                        cached_dataset.clear()
                        st.rerun()
            with clear_column:
                if st.button("Clear aid", width="stretch"):
                    save_garmin_aid_override(selected_aid_date, None)
                    cached_dataset.clear()
                    st.rerun()
            st.caption(
                "The selected aid applies from this date forward until another aid "
                "change is saved. Clear aid applies None from this date forward. "
                "Nothing is written back to Garmin Connect."
            )
    aid_score_column, aid_duration_column = st.columns(2)
    with aid_score_column:
        st.plotly_chart(
            aid_score_figure(view.selected.sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    with aid_duration_column:
        st.plotly_chart(
            aid_figure(view.selected.sessions),
            width="stretch",
            config={"displaylogo": False},
        )
    stage_share_column, stage_hours_column = st.columns(2)
    with stage_share_column:
        st.plotly_chart(
            aid_sleep_stage_composition_figure(
                view.selected.sessions, view.selected.sleep_stages
            ),
            width="stretch",
            config={"displaylogo": False},
        )
    with stage_hours_column:
        st.plotly_chart(
            aid_sleep_stage_composition_figure(
                view.selected.sessions,
                view.selected.sleep_stages,
                use_stage_duration=True,
            ),
            width="stretch",
            config={"displaylogo": False},
        )
    st.plotly_chart(
        aid_duration_timeline_figure(
            view.history.daily,
            view.history.sessions,
            view_start=view.start_date,
            view_end=view.end_date,
        ),
        width="stretch",
        config={"displaylogo": False},
    )
    st.caption(
        "Garmin Sleep scores are shown for night sleep only. Point colors indicate "
        "the recorded sleep aid. Aid comparisons are descriptive, not causal."
    )
