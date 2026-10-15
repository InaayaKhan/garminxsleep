"""Charts on the Patterns tab: duration distributions, sleep vs naps, sleep calendars."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from dashboard.charts.common import (
    NAP_BAR_COLOR,
    NIGHT_BAR_COLOR,
    SLEEP_SCORE_PINK,
    WEEKDAY_ORDER,
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
                    line=dict(color="rgba(14, 17, 23, 0.9)", width=1),
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
                    line=dict(color="rgba(14, 17, 23, 0.9)", width=1),
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


def calendar_sleep_heatmap(daily: pd.DataFrame) -> go.Figure:
    frame = daily.copy()
    frame["timestamp"] = pd.to_datetime(frame["date"])
    frame["week_start"] = frame["timestamp"] - pd.to_timedelta(
        frame["timestamp"].dt.dayofweek, unit="D"
    )
    frame["weekday_number"] = frame["timestamp"].dt.dayofweek
    frame["sleep_hours"] = frame["total_sleep_minutes"] / 60
    weeks = sorted(frame["week_start"].drop_duplicates())
    week_lookup = {week: index for index, week in enumerate(weeks)}
    frame["week_number"] = frame["week_start"].map(week_lookup)
    frame["weekday"] = frame["weekday_number"].map(dict(enumerate(WEEKDAY_ORDER)))
    frame["date_text"] = frame["timestamp"].dt.strftime("%Y-%m-%d")

    figure = go.Figure(
        go.Scatter(
            x=frame["week_number"],
            y=frame["weekday"],
            mode="markers",
            marker=dict(
                symbol="square",
                size=16,
                color=frame["sleep_hours"],
                cmin=0,
                cmax=max(12.0, float(frame["sleep_hours"].max())),
                colorscale=[
                    [0.0, "#111b2e"],
                    [0.5, "#145DA0"],
                    [1.0, NAP_BAR_COLOR],
                ],
                line=dict(color="rgba(255, 255, 255, 0.2)", width=1.5),
                colorbar=dict(title="Hours", thickness=12),
            ),
            customdata=frame[["date_text", "sleep_hours"]],
            hovertemplate=(
                "%{customdata[0]}<br>Sleep: %{customdata[1]:.2f} h<extra></extra>"
            ),
            showlegend=False,
        )
    )
    tick_positions = list(range(0, len(weeks), max(1, len(weeks) // 10 or 1)))
    figure.update_layout(
        title="Daily sleep duration calendar",
        height=330,
        margin=dict(l=80, r=65, t=64, b=56),
        xaxis=dict(
            title="Week starting",
            tickmode="array",
            tickvals=tick_positions,
            ticktext=[pd.Timestamp(weeks[index]).strftime("%d/%m") for index in tick_positions],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Day of week",
            categoryorder="array",
            categoryarray=WEEKDAY_ORDER,
            autorange="reversed",
            showgrid=False,
        ),
    )
    return figure


def calendar_sleep_score_heatmap(
    daily: pd.DataFrame, sessions: pd.DataFrame
) -> go.Figure:
    frame = daily[["date"]].copy()
    scored_nights = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & sessions["legacy_garmin_score_10"].notna()
    ]
    scores = (
        scored_nights.groupby("sleep_date")["legacy_garmin_score_10"].mean() * 10
    )
    frame["score"] = frame["date"].map(scores)
    frame["timestamp"] = pd.to_datetime(frame["date"])
    frame["week_start"] = frame["timestamp"] - pd.to_timedelta(
        frame["timestamp"].dt.dayofweek, unit="D"
    )
    frame["weekday_number"] = frame["timestamp"].dt.dayofweek
    weeks = sorted(frame["week_start"].drop_duplicates())
    week_lookup = {week: index for index, week in enumerate(weeks)}
    frame["week_number"] = frame["week_start"].map(week_lookup)
    frame["weekday"] = frame["weekday_number"].map(dict(enumerate(WEEKDAY_ORDER)))
    frame["date_text"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
    scored_frame = frame[frame["score"].notna()]

    figure = go.Figure(
        go.Scatter(
            x=scored_frame["week_number"],
            y=scored_frame["weekday"],
            mode="markers",
            marker=dict(
                symbol="square",
                size=16,
                color=scored_frame["score"],
                cmin=0,
                cmax=100,
                colorscale=[
                    [0.0, "#2A1721"],
                    [0.5, "#9C496F"],
                    [1.0, SLEEP_SCORE_PINK],
                ],
                line=dict(color="rgba(255, 255, 255, 0.2)", width=1.5),
                colorbar=dict(title="Score", thickness=12),
            ),
            customdata=scored_frame[["date_text", "score"]],
            hovertemplate=(
                "%{customdata[0]}<br>Garmin Sleep score: "
                "%{customdata[1]:.0f}/100<extra></extra>"
            ),
            showlegend=False,
        )
    )
    tick_positions = list(range(0, len(weeks), max(1, len(weeks) // 10 or 1)))
    figure.update_layout(
        title="Garmin Sleep score calendar",
        height=330,
        margin=dict(l=80, r=65, t=64, b=56),
        xaxis=dict(
            title="Week starting",
            tickmode="array",
            tickvals=tick_positions,
            ticktext=[pd.Timestamp(weeks[index]).strftime("%d/%m") for index in tick_positions],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Day of week",
            categoryorder="array",
            categoryarray=WEEKDAY_ORDER,
            autorange="reversed",
            showgrid=False,
        ),
    )
    return figure
