"""Charts on the Patterns tab: duration distributions and sleep vs naps."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from dashboard.charts.common import (
    NAP_BAR_COLOR,
    NIGHT_BAR_COLOR,
)


AVERAGE_3_COLOR = "#E8793F"


def _duration_histogram(
    values: pd.Series, *, title: str, color: str
) -> go.Figure:
    hours = pd.Series(values, dtype="float64").dropna()
    upper_bound = max(1, math.ceil(float(hours.max()))) if not hours.empty else 1
    figure = go.Figure(
        go.Histogram(
            x=hours,
            xbins=dict(start=0, end=upper_bound + 1, size=1),
            marker=dict(
                color=color,
                line=dict(color="rgba(14, 17, 23, 0.9)", width=0.7),
            ),
            hovertemplate=(
                "Duration bin: %{x:.0f} h<br>Observations: %{y}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=title,
        height=460,
        margin=dict(l=60, r=24, t=64, b=64),
        bargap=0.08,
        xaxis=dict(
            title="Sleep duration (hours)",
            range=[0, upper_bound + 1],
            dtick=2,
            zeroline=False,
        ),
        yaxis=dict(title="Number of observations", rangemode="tozero"),
    )
    return figure


def nightly_duration_histogram(sessions: pd.DataFrame) -> go.Figure:
    night_sleep = sessions[sessions["session_type"] == "NIGHT_SLEEP"]
    return _duration_histogram(
        night_sleep["sleep_minutes"] / 60,
        title="Nightly sleep duration distribution (naps excluded)",
        color=NIGHT_BAR_COLOR,
    )


def daily_duration_histogram(daily: pd.DataFrame) -> go.Figure:
    return _duration_histogram(
        daily["total_sleep_minutes"] / 60,
        title="Daily sleep duration distribution (naps included)",
        color=NAP_BAR_COLOR,
    )


def sleep_state_pie_figure(
    sessions: pd.DataFrame, daily: pd.DataFrame
) -> go.Figure:
    total_minutes = float(len(daily) * 24 * 60)
    night_minutes = float(
        sessions.loc[
            sessions["session_type"] == "NIGHT_SLEEP", "sleep_minutes"
        ].sum()
    )
    nap_minutes = float(
        sessions.loc[sessions["session_type"] == "NAP", "sleep_minutes"].sum()
    )
    hole_minutes = float(sessions["interruption_minutes"].sum())
    awake_minutes = max(
        0.0,
        total_minutes - night_minutes - nap_minutes - hole_minutes,
    )

    figure = go.Figure()
    if total_minutes == 0:
        figure.add_annotation(
            text="No calendar days in the selected date range",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    else:
        labels = ["Sleep", "Nap", "Awake", "Awake in holes"]
        values = [night_minutes, nap_minutes, awake_minutes, hole_minutes]
        figure.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                sort=False,
                direction="clockwise",
                marker=dict(
                    colors=[NIGHT_BAR_COLOR, NAP_BAR_COLOR, "#64748b", AVERAGE_3_COLOR],
                    line=dict(width=0),
                ),
                textinfo="label+percent",
                customdata=[[value / 60] for value in values],
                hovertemplate=(
                    "%{label}<br>%{customdata[0]:.2f} hours<br>"
                    "%{percent}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Time breakdown across selected days",
        height=520,
        margin=dict(l=40, r=40, t=64, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, x=0),
    )
    return figure


def sleep_nap_pie_figure(sessions: pd.DataFrame) -> go.Figure:
    night_minutes = float(
        sessions.loc[
            sessions["session_type"] == "NIGHT_SLEEP", "sleep_minutes"
        ].sum()
    )
    nap_minutes = float(
        sessions.loc[sessions["session_type"] == "NAP", "sleep_minutes"].sum()
    )
    total_sleep_minutes = night_minutes + nap_minutes

    figure = go.Figure()
    if total_sleep_minutes == 0:
        figure.add_annotation(
            text="No sleep in the selected date range",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    else:
        values = [night_minutes, nap_minutes]
        figure.add_trace(
            go.Pie(
                labels=["Sleep", "Nap"],
                values=values,
                sort=False,
                direction="clockwise",
                marker=dict(
                    colors=[NIGHT_BAR_COLOR, NAP_BAR_COLOR],
                    line=dict(width=0),
                ),
                textinfo="label+percent",
                customdata=[[value / 60] for value in values],
                hovertemplate=(
                    "%{label}<br>%{customdata[0]:.2f} hours<br>"
                    "%{percent}<extra></extra>"
                ),
            )
        )
    figure.update_layout(
        title="Sleep vs naps",
        height=520,
        margin=dict(l=40, r=40, t=64, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, x=0),
    )
    return figure
