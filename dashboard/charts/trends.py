"""Charts on the Trends & distributions tab."""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from dashboard.charts.common import (
    NAP_BAR_COLOR,
    NIGHT_BAR_COLOR,
    SLEEP_SCORE_PINK,
    STATS_RED,
    TARGET_SLEEP_HOURS,
    WEEKDAY_ORDER,
    _date_axis_limits,
    _linear_trend,
)


AVERAGE_3_COLOR = "#E8793F"
CREDIT_COLOR = "#E6B84F"


def _format_signed_duration(hours: float) -> str:
    total_minutes = int(round(abs(float(hours)) * 60))
    sign = "-" if hours < 0 else "+" if hours > 0 else ""
    return f"{sign}{total_minutes // 60}:{total_minutes % 60:02d}"


def _nice_tick_step(span: float, target_ticks: int = 5) -> float:
    rough_step = max(span / target_ticks, 1.0)
    magnitude = 10 ** math.floor(math.log10(rough_step))
    normalized = rough_step / magnitude
    for candidate in (1, 2, 5, 10):
        if normalized <= candidate:
            return candidate * magnitude
    return 10 * magnitude


def credit_debit_figure(
    daily: pd.DataFrame, target_sleep_hours: float = TARGET_SLEEP_HOURS
) -> go.Figure:
    """Reproduce Sleepmeter's cumulative credit/debit view.

    A target-duration day is neutral. Daily balance is scaled to a 24-hour
    credit/debit clock, so a zero-sleep day contributes -24:00.
    """
    frame = daily.sort_values("date").copy()
    frame["sleep_hours"] = frame["total_sleep_minutes"] / 60
    frame = frame[frame["sleep_hours"].notna()].copy()
    if frame.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="No completed sleep data in the selected date range",
            showarrow=False,
        )
        figure.update_layout(title="Sleep credit/debit", height=540)
        return figure
    frame["daily_balance"] = (
        frame["sleep_hours"] / target_sleep_hours - 1
    ) * 24
    frame["cumulative_balance"] = frame["daily_balance"].cumsum()
    frame["trend"] = _linear_trend(frame["cumulative_balance"])
    x_start, x_end = _date_axis_limits(frame["date"])

    mean_balance = float(frame["cumulative_balance"].mean())
    balance_std_value = frame["cumulative_balance"].std(ddof=1)
    balance_std = float(balance_std_value) if pd.notna(balance_std_value) else 0.0
    band_low = mean_balance - balance_std
    band_high = mean_balance + balance_std

    figure = go.Figure()
    figure.add_hrect(
        y0=band_low,
        y1=band_high,
        fillcolor=CREDIT_COLOR,
        opacity=0.14,
        line_width=0,
        layer="below",
    )
    figure.add_hline(y=0, line=dict(color=STATS_RED, width=2))
    figure.add_hline(
        y=mean_balance,
        line=dict(color=CREDIT_COLOR, width=1.5, dash="dash"),
    )
    figure.add_trace(
        go.Scatter(
            name="Credit/debit",
            x=frame["date"],
            y=frame["cumulative_balance"],
            mode="lines",
            line=dict(color=CREDIT_COLOR, width=1.8),
            customdata=frame[["sleep_hours", "daily_balance"]],
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>Sleep: %{customdata[0]:.2f} h<br>"
                "Daily change: %{customdata[1]:+.2f} h<br>"
                "Cumulative: %{y:+.2f} h<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Trend",
            x=frame["date"],
            y=frame["trend"],
            mode="lines",
            line=dict(color=CREDIT_COLOR, width=2.2),
            hoverinfo="skip",
        )
    )

    y_low = min(float(frame["cumulative_balance"].min()), band_low, 0.0)
    y_high = max(float(frame["cumulative_balance"].max()), band_high, 0.0)
    padding = max((y_high - y_low) * 0.04, 4.0)
    y_low -= padding
    y_high += padding
    tick_step = _nice_tick_step(y_high - y_low)
    tick_start = math.ceil(y_low / tick_step) * tick_step
    tick_values: list[float] = []
    value = tick_start
    while value <= y_high + tick_step * 0.01:
        tick_values.append(value)
        value += tick_step

    figure.update_layout(
        title="Sleep credit/debit",
        showlegend=False,
        height=540,
        margin=dict(l=90, r=24, t=64, b=64),
        hovermode="x",
        xaxis=dict(
            title="Date",
            type="date",
            range=[x_start, x_end],
            minallowed=x_start,
            maxallowed=x_end,
            tickformat="%d/%m",
            gridcolor="#303030",
        ),
        yaxis=dict(
            title="Credit/debit (+/− HH:MM)",
            range=[y_low, y_high],
            minallowed=y_low,
            maxallowed=y_high,
            tickmode="array",
            tickvals=tick_values,
            ticktext=[_format_signed_duration(item) for item in tick_values],
            gridcolor="#303030",
            zeroline=False,
        ),
    )
    return figure


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


def calendar_activity_heatmap(
    daily: pd.DataFrame, activities: pd.DataFrame
) -> go.Figure:
    calendar = daily[["date"]].copy()
    calendar["timestamp"] = pd.to_datetime(calendar["date"])
    calendar["week_start"] = calendar["timestamp"] - pd.to_timedelta(
        calendar["timestamp"].dt.dayofweek, unit="D"
    )
    calendar["weekday_number"] = calendar["timestamp"].dt.dayofweek
    weeks = sorted(calendar["week_start"].drop_duplicates())
    week_lookup = {week: index for index, week in enumerate(weeks)}

    if activities.empty:
        activity_days = pd.DataFrame()
    else:
        activity_days = (
            activities.groupby("activity_date", as_index=False)
            .agg(
                activity_type=(
                    "activity_type",
                    lambda values: (
                        str(values.iloc[0])
                        if values.nunique() == 1
                        else "multiple_activities"
                    ),
                ),
                activity_names=(
                    "activity_name",
                    lambda values: ", ".join(dict.fromkeys(map(str, values))),
                ),
                activity_minutes=(
                    "duration_seconds", lambda values: values.sum() / 60
                ),
            )
            .rename(columns={"activity_date": "date"})
        )
        activity_days = activity_days.merge(
            calendar[["date", "timestamp", "week_start", "weekday_number"]],
            on="date",
            how="inner",
        )
        activity_days["week_number"] = activity_days["week_start"].map(week_lookup)
        activity_days["weekday"] = activity_days["weekday_number"].map(
            dict(enumerate(WEEKDAY_ORDER))
        )
        activity_days["date_text"] = activity_days["timestamp"].dt.strftime("%Y-%m-%d")

    figure = go.Figure()
    if not activity_days.empty:
        figure.add_trace(
            go.Scatter(
                x=activity_days["week_number"],
                y=activity_days["weekday"],
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=16,
                    color=activity_days["activity_minutes"],
                    cmin=0,
                    cmax=float(activity_days["activity_minutes"].max()),
                    colorscale=[
                        [0.0, "#17251F"],
                        [0.5, "#257A59"],
                        [1.0, "#2FD6A3"],
                    ],
                    line=dict(color="rgba(255,255,255,0.25)", width=1.5),
                    colorbar=dict(title="Minutes", thickness=12),
                ),
                customdata=activity_days[
                    ["date_text", "activity_names", "activity_minutes"]
                ],
                hovertemplate=(
                    "%{customdata[0]}<br>%{customdata[1]}<br>"
                    "%{customdata[2]:.0f} minutes<extra></extra>"
                ),
                showlegend=False,
            )
        )
    tick_positions = list(range(0, len(weeks), max(1, len(weeks) // 10 or 1)))
    figure.update_layout(
        title="Recorded activity calendar",
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


def calendar_intensity_minutes_heatmap(
    daily: pd.DataFrame, metrics: pd.DataFrame
) -> go.Figure:
    frame = daily[["date"]].copy()
    intensity = (
        metrics.set_index("date")["daily_intensity_minutes"]
        if not metrics.empty and "daily_intensity_minutes" in metrics
        else pd.Series(dtype=float)
    )
    frame["intensity_minutes"] = frame["date"].map(intensity)
    frame = frame[frame["intensity_minutes"].notna()].copy()
    frame["timestamp"] = pd.to_datetime(frame["date"])
    frame["week_start"] = frame["timestamp"] - pd.to_timedelta(
        frame["timestamp"].dt.dayofweek, unit="D"
    )
    frame["weekday_number"] = frame["timestamp"].dt.dayofweek

    calendar_dates = daily[["date"]].copy()
    calendar_dates["timestamp"] = pd.to_datetime(calendar_dates["date"])
    calendar_dates["week_start"] = calendar_dates["timestamp"] - pd.to_timedelta(
        calendar_dates["timestamp"].dt.dayofweek, unit="D"
    )
    weeks = sorted(calendar_dates["week_start"].drop_duplicates())
    week_lookup = {week: index for index, week in enumerate(weeks)}
    frame["week_number"] = frame["week_start"].map(week_lookup)
    frame["weekday"] = frame["weekday_number"].map(dict(enumerate(WEEKDAY_ORDER)))
    frame["date_text"] = frame["timestamp"].dt.strftime("%Y-%m-%d")

    figure = go.Figure()
    if not frame.empty:
        figure.add_trace(
            go.Scatter(
                x=frame["week_number"],
                y=frame["weekday"],
                mode="markers",
                marker=dict(
                    symbol="square",
                    size=16,
                    color=frame["intensity_minutes"],
                    cmin=0,
                    cmax=max(1.0, float(frame["intensity_minutes"].max())),
                    colorscale=[
                        [0.0, "#17251F"],
                        [0.5, "#257A59"],
                        [1.0, "#2FD6A3"],
                    ],
                    line=dict(color="rgba(255,255,255,0.25)", width=1.5),
                    colorbar=dict(title="Minutes", thickness=12),
                ),
                customdata=frame[["date_text", "intensity_minutes"]],
                hovertemplate=(
                    "%{customdata[0]}<br>Intensity minutes: "
                    "%{customdata[1]:.0f}<extra></extra>"
                ),
                showlegend=False,
            )
        )
    tick_positions = list(range(0, len(weeks), max(1, len(weeks) // 10 or 1)))
    figure.update_layout(
        title="Intensity minutes calendar",
        height=330,
        margin=dict(l=80, r=65, t=64, b=56),
        xaxis=dict(
            title="Week starting",
            tickmode="array",
            tickvals=tick_positions,
            ticktext=[
                pd.Timestamp(weeks[index]).strftime("%d/%m")
                for index in tick_positions
            ],
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
