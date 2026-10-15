"""Charts on the Relationships tab."""

from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.charts.common import (
    NIGHT_BAR_COLOR,
    SLEEP_SCORE_PINK,
    STATS_RED,
    TARGET_SLEEP_HOURS,
    _clock_hour,
    _spearman_summary,
)


def _add_spearman_annotation(
    figure: go.Figure, x_values: pd.Series, y_values: pd.Series
) -> None:
    summary = _spearman_summary(x_values, y_values)
    if summary is None:
        return
    correlation, count = summary
    figure.add_annotation(
        x=0.99,
        y=0.99,
        xref="paper",
        yref="paper",
        text=f"Spearman ρ={correlation:+.2f} · n={count}",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(14,17,23,0.72)",
        borderpad=4,
        font=dict(size=11, color="#d1d5db"),
    )


def _format_clock_hour(hours: float) -> str:
    total_minutes = int(round(float(hours) * 60)) % (24 * 60)
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def menstrual_phase_comparisons_figure(
    daily: pd.DataFrame,
    sessions: pd.DataFrame,
    metrics: pd.DataFrame,
    menstrual: pd.DataFrame,
) -> go.Figure:
    """Compare sleep and recovery measurements across Garmin cycle phases."""
    panels = (
        ("Sleep duration", "sleep_hours", "hours"),
        ("Garmin Sleep score", "sleep_score", "score"),
        ("Overnight HRV", "overnight_hrv_display", "ms"),
        ("Resting heart rate", "daily_resting_heart_rate", "bpm"),
        ("Sleep stress", "average_sleep_stress", ""),
        ("REM share", "score_rem_percentage", "%"),
    )
    figure = make_subplots(
        rows=2,
        cols=3,
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
        subplot_titles=[panel[0] for panel in panels],
    )
    if menstrual.empty:
        figure.add_annotation(
            text="No menstrual phase data in the selected date range",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
    else:
        frame = menstrual[["date", "current_phase"]].copy()
        frame = frame.merge(
            daily[["date", "total_sleep_minutes"]], on="date", how="left"
        )
        frame["sleep_hours"] = frame["total_sleep_minutes"] / 60
        scored = sessions[
            (sessions["session_type"] == "NIGHT_SLEEP")
            & sessions["legacy_garmin_score_10"].notna()
        ].groupby("sleep_date")["legacy_garmin_score_10"].mean() * 10
        frame["sleep_score"] = frame["date"].map(scored)
        if not metrics.empty:
            metric_frame = metrics.copy()
            metric_frame["overnight_hrv_display"] = metric_frame[
                "hrv_last_night_average"
            ].fillna(metric_frame["overnight_hrv"])
            metric_columns = [
                "date",
                "overnight_hrv_display",
                "daily_resting_heart_rate",
                "average_sleep_stress",
                "score_rem_percentage",
            ]
            frame = frame.merge(metric_frame[metric_columns], on="date", how="left")
        phase_names = {1: "Menstrual", 2: "Follicular", 3: "Ovulation", 4: "Luteal"}
        phase_colors = {
            "Menstrual": "#E14E5C",
            "Follicular": "#2FD6A3",
            "Ovulation": "#FFB547",
            "Luteal": "#F29BDA",
        }
        frame["phase"] = frame["current_phase"].map(phase_names)
        for panel_index, (title, column, unit) in enumerate(panels):
            row, col = divmod(panel_index, 3)
            row += 1
            col += 1
            if column not in frame:
                continue
            for phase in phase_names.values():
                values = frame.loc[frame["phase"] == phase, column].dropna()
                if values.empty:
                    continue
                figure.add_trace(
                    go.Box(
                        name=phase,
                        x=[phase] * len(values),
                        y=values,
                        marker_color=phase_colors[phase],
                        fillcolor="rgba(0,0,0,0)",
                        line=dict(color=phase_colors[phase], width=1.3),
                        boxmean=True,
                        showlegend=panel_index == 0,
                        legendgroup=phase,
                        hovertemplate=(
                            f"{phase}<br>{title}: %{{y:.1f}} {unit}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
            figure.update_yaxes(title_text=unit, row=row, col=col)
    figure.update_layout(
        title="Sleep and recovery by menstrual phase",
        height=690,
        margin=dict(l=62, r=28, t=92, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        boxmode="group",
    )
    figure.update_xaxes(showgrid=False)
    return figure


def multifactor_relationships_figure(
    daily: pd.DataFrame,
    sessions: pd.DataFrame,
    activities: pd.DataFrame,
    metrics: pd.DataFrame,
    menstrual: pd.DataFrame,
) -> go.Figure:
    frame = daily[["date", "total_sleep_minutes"]].sort_values("date").copy()
    frame["sleep_hours"] = frame["total_sleep_minutes"] / 60
    prior_deficit = (TARGET_SLEEP_HOURS - frame["sleep_hours"]).shift(1)
    frame["sleep_debt_3"] = prior_deficit.rolling(3, min_periods=1).sum()
    frame["sleep_debt_7"] = prior_deficit.rolling(7, min_periods=1).sum()
    frame["weekend"] = pd.to_datetime(frame["date"]).dt.dayofweek.isin([5, 6]).astype(float)
    scores = sessions[(sessions["session_type"] == "NIGHT_SLEEP") & sessions["legacy_garmin_score_10"].notna()].groupby("sleep_date")["legacy_garmin_score_10"].mean() * 10
    frame["sleep_score"] = frame["date"].map(scores)
    activity_hours = activities.groupby("activity_date")["duration_seconds"].sum() / 3600 if not activities.empty else pd.Series(dtype=float)
    frame["previous_activity"] = frame["date"].map(lambda value: activity_hours.get(value - timedelta(days=1), 0.0))
    if not menstrual.empty:
        frame = frame.merge(menstrual[["date", "current_phase"]], on="date", how="left")
    else:
        frame["current_phase"] = float("nan")
    if not metrics.empty:
        metric = metrics.copy()
        metric["hrv"] = metric["hrv_last_night_average"].fillna(metric["overnight_hrv"])
        frame = frame.merge(
            metric[["date", "hrv", "average_sleep_stress"]],
            on="date",
            how="left",
        )
        previous_metric_columns = {
            "daily_steps": "previous_steps",
            "daily_floors_ascended": "previous_floors",
            "daily_intensity_minutes": "previous_intensity",
            "daily_active_calories": "previous_active_calories",
            "daily_average_stress": "previous_daily_stress",
        }
        metric_by_date = metric.groupby("date").first()
        for source_column, target_column in previous_metric_columns.items():
            if source_column in metric_by_date:
                frame[target_column] = frame["date"].map(
                    lambda value, column=source_column: metric_by_date[column].get(
                        value - timedelta(days=1), float("nan")
                    )
                )
            else:
                frame[target_column] = float("nan")
    else:
        frame["hrv"] = float("nan")
        frame["average_sleep_stress"] = float("nan")
        for column in (
            "previous_steps",
            "previous_floors",
            "previous_intensity",
            "previous_active_calories",
            "previous_daily_stress",
        ):
            frame[column] = float("nan")
    predictors = [
        "sleep_hours",
        "sleep_debt_3",
        "sleep_debt_7",
        "previous_activity",
        "previous_steps",
        "previous_floors",
        "previous_intensity",
        "previous_active_calories",
        "previous_daily_stress",
        "weekend",
    ]
    phase_dummies = pd.get_dummies(frame["current_phase"], prefix="phase", dtype=float)
    design = pd.concat([frame[predictors], phase_dummies], axis=1)
    labels = {
        "sleep_hours": "Sleep duration",
        "sleep_debt_3": "3-day sleep debt",
        "sleep_debt_7": "7-day sleep debt",
        "previous_activity": "Previous-day activity",
        "previous_steps": "Previous-day steps",
        "previous_floors": "Previous-day floors climbed",
        "previous_intensity": "Previous-day intensity minutes",
        "previous_active_calories": "Previous-day active calories",
        "previous_daily_stress": "Previous-day stress",
        "weekend": "Weekend",
    }
    phase_labels = {
        1: "Menstrual phase",
        2: "Follicular phase",
        3: "Ovulation phase",
        4: "Luteal phase",
    }
    for column in phase_dummies:
        try:
            phase_number = int(float(column.removeprefix("phase_")))
        except ValueError:
            phase_number = 0
        labels[column] = phase_labels.get(phase_number, column)
    outcomes = (
        ("sleep_score", "Sleep score", 1, 1),
        ("hrv", "Overnight HRV", 1, 2),
        ("average_sleep_stress", "Average sleep stress", 2, 1),
        ("sleep_hours", "Sleep duration", 2, 2),
    )
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[item[1] for item in outcomes],
        vertical_spacing=0.13,
    )
    for outcome, _title, row_index, column_index in outcomes:
        outcome_predictors = [
            column
            for column in design
            if not (outcome == "sleep_hours" and column == "sleep_hours")
        ]
        combined = pd.concat(
            [design[outcome_predictors], frame[[outcome]]], axis=1
        ).dropna()
        usable = [
            column
            for column in outcome_predictors
            if combined[column].nunique() > 1
        ]
        if len(combined) < len(usable) + 3 or not usable:
            continue
        x = combined[usable].astype(float)
        x = (x - x.mean()) / x.std(ddof=0).replace(0, 1)
        y = combined[outcome].astype(float)
        y = (y - y.mean()) / (y.std(ddof=0) or 1)
        coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0][1:]
        order = np.argsort(np.abs(coefficients))
        figure.add_trace(go.Bar(x=coefficients[order], y=[labels[c] for c in np.array(usable)[order]], orientation="h", marker_color=["#4DB88A" if value >= 0 else "#E14E5C" for value in coefficients[order]], showlegend=False), row=row_index, col=column_index)
        figure.update_xaxes(title_text="Standardized coefficient", zeroline=True, row=row_index, col=column_index)
    figure.update_layout(title="Multifactor relationships", height=980, margin=dict(l=180, r=30, t=90, b=60))
    return figure


def nightly_duration_quality_figure(
    sessions: pd.DataFrame, target_sleep_hours: float = TARGET_SLEEP_HOURS
) -> go.Figure:
    frame = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & sessions["legacy_garmin_score_10"].notna()
    ].copy()
    frame["sleep_hours"] = frame["sleep_minutes"] / 60
    frame["garmin_score_100"] = frame["legacy_garmin_score_10"] * 10
    frame = frame.sort_values("sleep_hours")

    if frame.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="No scored night sleep in the selected date range",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        figure.update_layout(
            title="Nightly duration vs Sleep score",
            height=540,
            margin=dict(l=80, r=24, t=64, b=72),
            xaxis=dict(
                title="Nightly sleep duration (HH:MM)",
                range=[0, 12],
                minallowed=0,
                maxallowed=12,
            ),
            yaxis=dict(
                title="Sleep score (0–100)",
                range=[0, 100],
                minallowed=0,
                maxallowed=100,
                dtick=10,
            ),
        )
        return figure

    quality_mean = float(frame["garmin_score_100"].mean())
    quality_std_value = frame["garmin_score_100"].std(ddof=1)
    quality_std = float(quality_std_value) if pd.notna(quality_std_value) else 0.0
    centered = frame["sleep_hours"] - frame["sleep_hours"].mean()
    denominator = float((centered**2).sum())
    slope = (
        float(
            (
                centered
                * (frame["garmin_score_100"] - quality_mean)
            ).sum()
            / denominator
        )
        if denominator
        else 0.0
    )
    intercept = quality_mean - slope * float(frame["sleep_hours"].mean())
    x_max = max(12.0, math.ceil(float(frame["sleep_hours"].max())))
    trend_x = [0.0, x_max]
    trend_y = [intercept, intercept + slope * x_max]

    figure = go.Figure()
    figure.add_hrect(
        y0=max(0.0, quality_mean - quality_std),
        y1=min(100.0, quality_mean + quality_std),
        fillcolor=SLEEP_SCORE_PINK,
        opacity=0.14,
        line_width=0,
        layer="below",
    )
    figure.add_hline(
        y=quality_mean,
        line=dict(color=SLEEP_SCORE_PINK, width=1.5, dash="dash"),
    )
    figure.add_vline(
        x=target_sleep_hours,
        line=dict(color=STATS_RED, width=2),
    )
    figure.add_trace(
        go.Scatter(
            name="Night sleep",
            x=frame["sleep_hours"],
            y=frame["garmin_score_100"],
            mode="markers",
            marker=dict(color=SLEEP_SCORE_PINK, size=5),
            customdata=frame[["sleep_date", "aid_name"]],
            hovertemplate=(
                "%{customdata[0]}<br>Duration: %{x:.2f} h<br>"
                "Score: %{y}/100<br>Aid: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            name="Trend",
            x=trend_x,
            y=trend_y,
            mode="lines",
            line=dict(color=SLEEP_SCORE_PINK, width=2.2),
            hoverinfo="skip",
        )
    )

    x_ticks = list(range(0, int(x_max) + 1, 2))
    figure.update_layout(
        title="Nightly duration vs Sleep score",
        showlegend=False,
        height=540,
        margin=dict(l=80, r=24, t=64, b=72),
        xaxis=dict(
            title="Nightly sleep duration (HH:MM)",
            range=[0, x_max],
            minallowed=0,
            maxallowed=x_max,
            tickmode="array",
            tickvals=x_ticks,
            ticktext=[f"{hour}:00" for hour in x_ticks],
            gridcolor="#303030",
        ),
        yaxis=dict(
            title="Sleep score (0–100)",
            range=[0, 100],
            minallowed=0,
            maxallowed=100,
            dtick=10,
            gridcolor="#303030",
            zeroline=False,
        ),
    )
    _add_spearman_annotation(
        figure, frame["sleep_hours"], frame["garmin_score_100"]
    )
    return figure


def asleep_time_vs_duration_figure(sessions: pd.DataFrame) -> go.Figure:
    frame = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (sessions["sleep_minutes"] > 0)
    ].copy()
    frame["asleep_hour"] = frame["sleep_start"].map(
        lambda value: _clock_hour(value, 12)
    )
    frame["sleep_hours"] = frame["sleep_minutes"] / 60
    frame = frame.sort_values("asleep_hour")

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            name="Night sleep",
            x=frame["asleep_hour"],
            y=frame["sleep_hours"],
            mode="markers",
            marker=dict(color=NIGHT_BAR_COLOR, size=7, opacity=0.75),
            customdata=frame[["sleep_date", "aid_name", "sleep_start"]],
            hovertemplate=(
                "%{customdata[0]}<br>Asleep: %{customdata[2]|%H:%M}<br>"
                "Duration: %{y:.2f} h<br>Aid: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    if len(frame) >= 2:
        centered = frame["asleep_hour"] - frame["asleep_hour"].mean()
        denominator = float((centered**2).sum())
        slope = (
            float(
                (
                    centered
                    * (frame["sleep_hours"] - frame["sleep_hours"].mean())
                ).sum()
                / denominator
            )
            if denominator
            else 0.0
        )
        intercept = float(frame["sleep_hours"].mean()) - slope * float(
            frame["asleep_hour"].mean()
        )
        x_start = float(frame["asleep_hour"].min())
        x_end = float(frame["asleep_hour"].max())
        figure.add_trace(
            go.Scatter(
                name="Trend",
                x=[x_start, x_end],
                y=[intercept + slope * x_start, intercept + slope * x_end],
                mode="lines",
                line=dict(color=NIGHT_BAR_COLOR, width=2.2),
                hoverinfo="skip",
            )
        )
    y_max = max(12.0, math.ceil(float(frame["sleep_hours"].max()))) if not frame.empty else 12.0
    figure.update_layout(
        title="Asleep time vs nightly sleep duration",
        height=540,
        margin=dict(l=70, r=24, t=64, b=64),
        showlegend=False,
        xaxis=dict(
            title="Asleep time",
            range=[12, 36],
            tickmode="array",
            tickvals=[13, 17, 21, 25, 29, 33],
            ticktext=["13:00", "17:00", "21:00", "01:00", "05:00", "09:00"],
            zeroline=False,
        ),
        yaxis=dict(
            title="Nightly sleep duration (hours)",
            range=[0, y_max],
            zeroline=False,
        ),
    )
    _add_spearman_annotation(figure, frame["asleep_hour"], frame["sleep_hours"])
    return figure


def sleep_efficiency_figure(sessions: pd.DataFrame) -> go.Figure:
    frame = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (sessions["gross_minutes"] > 0)
    ].copy()
    frame = frame.sort_values("sleep_date")
    frame["efficiency"] = frame["sleep_minutes"] / frame["gross_minutes"] * 100
    figure = go.Figure()
    if not frame.empty:
        mean = float(frame["efficiency"].mean())
        std = float(frame["efficiency"].std(ddof=1)) if len(frame) > 1 else 0.0
        figure.add_hrect(
            y0=max(0, mean - std),
            y1=min(100, mean + std),
            fillcolor=NIGHT_BAR_COLOR,
            opacity=0.14,
            line_width=0,
        )
        figure.add_hline(
            y=mean,
            line=dict(color=NIGHT_BAR_COLOR, width=1.5, dash="dash"),
        )
        figure.add_trace(
            go.Scatter(
                x=pd.to_datetime(frame["sleep_date"]),
                y=frame["efficiency"],
                mode="lines",
                line=dict(color=NIGHT_BAR_COLOR, width=1.8),
                customdata=frame[["interruption_minutes"]],
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>Efficiency: %{y:.1f}%<br>"
                    "Awake in holes: %{customdata[0]:.0f} min<extra></extra>"
                ),
                showlegend=False,
            )
        )
    figure.update_layout(
        title="Nightly sleep efficiency",
        height=500,
        margin=dict(l=70, r=24, t=64, b=64),
        xaxis=dict(title="Sleep date", tickformat="%d/%m"),
        yaxis=dict(title="Sleep efficiency", range=[0, 100], ticksuffix="%"),
    )
    return figure


def _add_scatter_trend(
    figure: go.Figure,
    x_values: pd.Series,
    y_values: pd.Series,
    *,
    color: str,
) -> None:
    if len(x_values) < 2:
        return
    x_numeric = pd.Series(x_values, dtype="float64").reset_index(drop=True)
    y_numeric = pd.Series(y_values, dtype="float64").reset_index(drop=True)
    centered = x_numeric - x_numeric.mean()
    denominator = float((centered**2).sum())
    slope = (
        float((centered * (y_numeric - y_numeric.mean())).sum() / denominator)
        if denominator
        else 0.0
    )
    intercept = float(y_numeric.mean()) - slope * float(x_numeric.mean())
    x_start = float(x_numeric.min())
    x_end = float(x_numeric.max())
    figure.add_trace(
        go.Scatter(
            name="Trend",
            x=[x_start, x_end],
            y=[intercept + slope * x_start, intercept + slope * x_end],
            mode="lines",
            line=dict(color=color, width=2.2),
            hoverinfo="skip",
            showlegend=False,
        )
    )


def asleep_time_vs_score_figure(sessions: pd.DataFrame) -> go.Figure:
    frame = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (sessions["sleep_minutes"] > 0)
        & sessions["legacy_garmin_score_10"].notna()
    ].copy()
    frame["asleep_hour"] = frame["sleep_start"].map(
        lambda value: _clock_hour(value, 12)
    )
    frame["asleep_text"] = frame["asleep_hour"].map(_format_clock_hour)
    frame["score_100"] = frame["legacy_garmin_score_10"] * 10
    figure = go.Figure(
        go.Scatter(
            x=frame["asleep_hour"],
            y=frame["score_100"],
            mode="markers",
            marker=dict(color=SLEEP_SCORE_PINK, size=7, opacity=0.75),
            customdata=frame[["sleep_date", "aid_name", "asleep_text"]],
            hovertemplate=(
                "%{customdata[0]}<br>Asleep: %{customdata[2]}<br>"
                "Garmin Sleep score: %{y:.0f}/100<br>"
                "Aid: %{customdata[1]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    _add_scatter_trend(
        figure,
        frame["asleep_hour"],
        frame["score_100"],
        color=SLEEP_SCORE_PINK,
    )
    figure.update_layout(
        title="Asleep time vs Garmin Sleep score",
        height=520,
        margin=dict(l=70, r=24, t=64, b=64),
        xaxis=dict(
            title="Asleep time",
            range=[12, 36],
            tickmode="array",
            tickvals=[13, 17, 21, 25, 29, 33],
            ticktext=["13:00", "17:00", "21:00", "01:00", "05:00", "09:00"],
        ),
        yaxis=dict(title="Garmin Sleep score (0–100)", range=[0, 100], dtick=10),
    )
    _add_spearman_annotation(figure, frame["asleep_hour"], frame["score_100"])
    return figure


def sleep_efficiency_vs_score_figure(sessions: pd.DataFrame) -> go.Figure:
    frame = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (sessions["gross_minutes"] > 0)
        & sessions["legacy_garmin_score_10"].notna()
    ].copy()
    frame["efficiency"] = frame["sleep_minutes"] / frame["gross_minutes"] * 100
    frame["score_100"] = frame["legacy_garmin_score_10"] * 10
    figure = go.Figure(
        go.Scatter(
            x=frame["efficiency"],
            y=frame["score_100"],
            mode="markers",
            marker=dict(color=SLEEP_SCORE_PINK, size=7, opacity=0.75),
            customdata=frame[["sleep_date", "interruption_minutes", "aid_name"]],
            hovertemplate=(
                "%{customdata[0]}<br>Sleep efficiency: %{x:.1f}%<br>"
                "Garmin Sleep score: %{y:.0f}/100<br>"
                "Awake in holes: %{customdata[1]:.0f} min<br>"
                "Aid: %{customdata[2]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    _add_scatter_trend(
        figure,
        frame["efficiency"],
        frame["score_100"],
        color=SLEEP_SCORE_PINK,
    )
    figure.update_layout(
        title="Sleep efficiency vs Garmin Sleep score",
        height=520,
        margin=dict(l=70, r=24, t=64, b=64),
        xaxis=dict(title="Sleep efficiency", range=[0, 100], ticksuffix="%"),
        yaxis=dict(title="Garmin Sleep score (0–100)", range=[0, 100], dtick=10),
    )
    _add_spearman_annotation(figure, frame["efficiency"], frame["score_100"])
    return figure
