"""Charts on the Timing tab: bed and wake times, weekday patterns."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go

from dashboard.charts.common import (
    AVERAGE_7_COLOR,
    NAP_BAR_COLOR,
    NIGHT_BAR_COLOR,
    SLEEP_SCORE_PINK,
    WEEKDAY_ORDER,
    _clock_hour,
    _date_axis_limits,
    _linear_trend,
)


ASLEEP_TIME_COLOR = "#5B32D6"
WAKE_TIME_COLOR = "#E8793F"


def _weekend_highlight_time_figure(
    sessions: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    yaxis_title: str,
    y_range: tuple[float, float],
    tick_values: list[float],
    tick_text: list[str],
    weekday_color: str,
    weekend_color: str,
    boundary_hour: int,
) -> go.Figure:
    frame = sessions[sessions["session_type"] == "NIGHT_SLEEP"].copy()
    frame = frame.sort_values("sleep_date")
    frame["timeline"] = pd.to_datetime(frame["sleep_date"])
    frame["clock_value"] = frame[value_column].map(
        lambda value: _clock_hour(value, boundary_hour)
    )
    frame["is_weekend"] = frame["timeline"].dt.dayofweek >= 5
    frame["day_group"] = frame["is_weekend"].map(
        {False: "Weekday", True: "Weekend"}
    )
    figure = go.Figure()
    if not frame.empty:
        mean = float(frame["clock_value"].mean())
        std_value = frame["clock_value"].std(ddof=1)
        std = float(std_value) if pd.notna(std_value) else 0.0
        figure.add_hrect(
            y0=max(y_range[0], mean - std),
            y1=min(y_range[1], mean + std),
            fillcolor=weekday_color,
            opacity=0.14,
            line_width=0,
            layer="below",
        )
        figure.add_hline(
            y=mean,
            line=dict(color=weekday_color, width=1.5, dash="dash"),
        )
    for index in range(len(frame) - 1):
        current = frame.iloc[index]
        following = frame.iloc[index + 1]
        segment_color = weekend_color if following["is_weekend"] else weekday_color
        figure.add_trace(
            go.Scatter(
                x=[current["timeline"], following["timeline"]],
                y=[current["clock_value"], following["clock_value"]],
                mode="lines",
                line=dict(color=segment_color, width=1.8),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    figure.add_trace(
        go.Scatter(
            name="Overall trend",
            x=frame["timeline"],
            y=_linear_trend(frame["clock_value"]),
            mode="lines",
            line=dict(color=weekday_color, width=2.2),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    xaxis: dict[str, object] = {"title": "Sleep date", "tickformat": "%d/%m"}
    if not frame.empty:
        x_start, x_end = _date_axis_limits(frame["timeline"])
        xaxis.update(
            type="date",
            range=[x_start, x_end],
            minallowed=x_start,
            maxallowed=x_end,
        )
    figure.update_layout(
        title=title,
        height=540,
        margin=dict(l=90, r=24, t=64, b=64),
        hovermode="closest",
        xaxis=xaxis,
        yaxis=dict(
            title=yaxis_title,
            range=list(y_range),
            minallowed=y_range[0],
            maxallowed=y_range[1],
            tickmode="array",
            tickvals=tick_values,
            ticktext=tick_text,
            zeroline=False,
        ),
    )
    return figure


def asleep_time_weekend_highlight_figure(sessions: pd.DataFrame) -> go.Figure:
    return _weekend_highlight_time_figure(
        sessions,
        value_column="sleep_start",
        title="Asleep time — weekends highlighted",
        yaxis_title="Asleep time",
        y_range=(12, 36),
        tick_values=[13, 17, 21, 25, 29, 33],
        tick_text=["13:00", "17:00", "21:00", "01:00", "05:00", "09:00"],
        weekday_color=ASLEEP_TIME_COLOR,
        weekend_color="#A78BFA",
        boundary_hour=12,
    )


def wake_up_time_weekend_highlight_figure(sessions: pd.DataFrame) -> go.Figure:
    return _weekend_highlight_time_figure(
        sessions,
        value_column="wake_time",
        title="Wake-up time — weekends highlighted",
        yaxis_title="Wake-up time",
        y_range=(0, 24),
        tick_values=[1, 5, 9, 13, 17, 21],
        tick_text=["01:00", "05:00", "09:00", "13:00", "17:00", "21:00"],
        weekday_color=WAKE_TIME_COLOR,
        weekend_color="#FDBA74",
        boundary_hour=0,
    )


def _duration_by_weekday_boxplot(
    dates: pd.Series,
    hours: pd.Series,
    *,
    title: str,
    color: str,
    fill_color: str,
    weekend_color: str,
    weekend_fill_color: str,
) -> go.Figure:
    frame = pd.DataFrame(
        {
            "weekday": pd.to_datetime(dates).dt.day_name(),
            "hours": pd.Series(hours, dtype="float64").to_numpy(),
        }
    )
    figure = go.Figure()
    for weekday in WEEKDAY_ORDER:
        subset = frame[frame["weekday"] == weekday]
        if subset.empty:
            continue
        is_weekend = weekday in {"Saturday", "Sunday"}
        box_color = weekend_color if is_weekend else color
        box_fill_color = weekend_fill_color if is_weekend else fill_color
        figure.add_trace(
            go.Box(
                name=weekday,
                x=[weekday] * len(subset),
                y=subset["hours"],
                boxpoints="all",
                jitter=0.25,
                pointpos=0,
                marker=dict(color=box_color, size=5),
                line_color=box_color,
                fillcolor=box_fill_color,
                hovertemplate=(
                    "%{x}<br>Duration: %{y:.2f} h<extra></extra>"
                ),
                showlegend=False,
            )
        )
    weekday_means = frame.groupby("weekday")["hours"].mean().reindex(WEEKDAY_ORDER)
    weekday_means = weekday_means.dropna()
    figure.add_trace(
        go.Scatter(
            name="Mean",
            x=weekday_means.index,
            y=weekday_means.values,
            mode="markers",
            marker=dict(
                color=AVERAGE_7_COLOR,
                symbol="diamond",
                size=5,
                line=dict(color="#F8DC7A", width=1),
            ),
            hovertemplate="%{x}<br>Mean: %{y:.2f} h<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        height=520,
        margin=dict(l=60, r=24, t=64, b=64),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(
            title="Day of week",
            categoryorder="array",
            categoryarray=WEEKDAY_ORDER,
        ),
        yaxis=dict(title="Sleep duration (hours)", rangemode="tozero"),
    )
    return figure


def nightly_duration_by_weekday_boxplot(sessions: pd.DataFrame) -> go.Figure:
    night_sleep = sessions[sessions["session_type"] == "NIGHT_SLEEP"]
    return _duration_by_weekday_boxplot(
        night_sleep["sleep_date"],
        night_sleep["sleep_minutes"] / 60,
        title="Nightly Duration by Day of Week — boxplots",
        color=NIGHT_BAR_COLOR,
        fill_color="rgba(91, 50, 214, 0.35)",
        weekend_color="#A78BFA",
        weekend_fill_color="rgba(167, 139, 250, 0.35)",
    )


def daily_duration_by_weekday_boxplot(daily: pd.DataFrame) -> go.Figure:
    return _duration_by_weekday_boxplot(
        daily["date"],
        daily["total_sleep_minutes"] / 60,
        title="Daily Duration by Day of Week — boxplots",
        color=NAP_BAR_COLOR,
        fill_color="rgba(8, 124, 255, 0.35)",
        weekend_color="#8FC5FF",
        weekend_fill_color="rgba(143, 197, 255, 0.35)",
    )


def hourly_sleep_punch_card(
    intervals: pd.DataFrame,
    *,
    title: str = "Hourly sleep punch card (night sleep + naps)",
    color: str = NAP_BAR_COLOR,
    light_color: str = "#8FC5FF",
) -> go.Figure:
    asleep_hours: set[tuple[date, int]] = set()
    for row in intervals[["interval_start", "interval_end"]].itertuples(index=False):
        cursor = pd.Timestamp(row.interval_start).floor("h")
        interval_end = pd.Timestamp(row.interval_end)
        while cursor < interval_end:
            hour_end = cursor + pd.Timedelta(hours=1)
            if max(cursor, pd.Timestamp(row.interval_start)) < min(hour_end, interval_end):
                asleep_hours.add((cursor.date(), cursor.hour))
            cursor = hour_end

    counts = pd.DataFrame(asleep_hours, columns=["calendar_date", "hour"])
    if counts.empty:
        summary = pd.DataFrame(columns=["weekday", "hour", "count"])
    else:
        counts["weekday"] = pd.to_datetime(counts["calendar_date"]).dt.day_name()
        summary = (
            counts.groupby(["weekday", "hour"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )

    max_count = int(summary["count"].max()) if not summary.empty else 1
    marker_sizes = 5 + (summary["count"] / max_count * 25) if not summary.empty else []
    figure = go.Figure(
        go.Scatter(
            x=summary["hour"],
            y=summary["weekday"],
            mode="markers",
            marker=dict(
                size=marker_sizes,
                color=summary["count"],
                colorscale=[[0, light_color], [1, color]],
                cmin=0,
                cmax=max_count,
                line=dict(color="rgba(255, 255, 255, 0.35)", width=0.6),
                colorbar=dict(title="Days asleep"),
            ),
            customdata=summary[["count"]],
            hovertemplate=(
                "%{y}<br>%{x}:00–%{x}:59<br>"
                "Days asleep: %{customdata[0]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.update_layout(
        title=title,
        height=500,
        margin=dict(l=90, r=80, t=64, b=64),
        xaxis=dict(
            title="Hour of day",
            range=[-0.5, 23.5],
            tickmode="array",
            tickvals=list(range(24)),
            ticktext=[f"{hour:02d}:00" for hour in range(24)],
            tickangle=-45,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Day of week",
            categoryorder="array",
            categoryarray=WEEKDAY_ORDER,
            autorange="reversed",
            showgrid=False,
            zeroline=False,
        ),
    )
    return figure


def score_by_weekday_boxplot(sessions: pd.DataFrame) -> go.Figure:
    frame = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & sessions["legacy_garmin_score_10"].notna()
    ].copy()
    frame["weekday"] = pd.to_datetime(frame["sleep_date"]).dt.day_name()
    frame["score_100"] = frame["legacy_garmin_score_10"] * 10

    figure = go.Figure()
    for weekday in WEEKDAY_ORDER:
        subset = frame[frame["weekday"] == weekday]
        if subset.empty:
            continue
        is_weekend = weekday in {"Saturday", "Sunday"}
        color = "#FFB3D1" if is_weekend else SLEEP_SCORE_PINK
        fill_color = (
            "rgba(255, 179, 209, 0.35)"
            if is_weekend
            else "rgba(255, 122, 182, 0.35)"
        )
        figure.add_trace(
            go.Box(
                name=weekday,
                x=[weekday] * len(subset),
                y=subset["score_100"],
                boxpoints="all",
                jitter=0.25,
                pointpos=0,
                marker=dict(color=color, size=5),
                line_color=color,
                fillcolor=fill_color,
                hovertemplate=(
                    "%{x}<br>Garmin Sleep score: %{y:.0f}/100<extra></extra>"
                ),
                showlegend=False,
            )
        )
    means = frame.groupby("weekday")["score_100"].mean().reindex(WEEKDAY_ORDER)
    means = means.dropna()
    figure.add_trace(
        go.Scatter(
            name="Mean",
            x=means.index,
            y=means.values,
            mode="markers",
            marker=dict(
                color=AVERAGE_7_COLOR,
                symbol="diamond",
                size=7,
                line=dict(color="#F8DC7A", width=1),
            ),
            hovertemplate=(
                "%{x}<br>Mean Garmin Sleep score: %{y:.1f}/100<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Garmin Sleep score by day of week",
        height=520,
        margin=dict(l=70, r=24, t=64, b=64),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(
            title="Day of week",
            categoryorder="array",
            categoryarray=WEEKDAY_ORDER,
        ),
        yaxis=dict(title="Garmin Sleep score (0–100)", range=[0, 100], dtick=10),
    )
    return figure
