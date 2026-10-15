"""Charts on the Activity & sleep tab: activity calendars and the following night."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.charts.common import (
    WEEKDAY_ORDER,
    _spearman_summary,
)
from dashboard.config import DAYTIME_WINDOW_MINUTES, DAYTIME_WINDOW_TEXT, TIMEZONE_NAME


def activity_effect_on_sleep_figure(
    activities: pd.DataFrame,
    sessions: pd.DataFrame,
    metrics: pd.DataFrame,
    predictor: str = "duration",
) -> go.Figure:
    """Relate each activity day's load to its next recorded night sleep."""
    panels = (
        ("Subsequent sleep duration", "sleep_hours", "hours", "#5B32D6"),
        ("Subsequent Sleep score", "sleep_score", "score", "#FF7AB6"),
        ("Subsequent overnight HRV", "overnight_hrv_display", "ms", "#4DB88A"),
        ("Subsequent sleep stress", "average_sleep_stress", "", "#E6B84F"),
    )
    figure = make_subplots(
        rows=2,
        cols=2,
        vertical_spacing=0.17,
        horizontal_spacing=0.10,
        subplot_titles=[panel[0] for panel in panels],
    )
    nights = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (~sessions["zero_duration"])
    ].sort_values("sleep_start")
    if activities.empty or nights.empty:
        figure.add_annotation(
            text="No activity-to-subsequent-sleep pairs in the selected date range",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
    else:
        daily_activity = (
            activities.groupby("activity_date", as_index=False)
            .agg(
                activity_seconds=("duration_seconds", "sum"),
                last_activity_end=("activity_end", "max"),
                activity_types=(
                    "activity_type",
                    lambda values: ", ".join(
                        sorted({str(value).replace("_", " ").title() for value in values})
                    ),
                ),
            )
            .sort_values("activity_date")
        )
        metric_frame = metrics.copy()
        if not metric_frame.empty:
            metric_frame["overnight_hrv_display"] = metric_frame[
                "hrv_last_night_average"
            ].fillna(metric_frame["overnight_hrv"])
            metric_frame = metric_frame.set_index("date")
        pairs: list[dict[str, object]] = []
        for activity in daily_activity.itertuples(index=False):
            next_nights = nights[nights["sleep_start"] >= activity.last_activity_end]
            if next_nights.empty:
                continue
            night = next_nights.iloc[0]
            sleep_date = night["sleep_date"]
            metric = (
                metric_frame.loc[sleep_date]
                if not metric_frame.empty and sleep_date in metric_frame.index
                else None
            )
            if isinstance(metric, pd.DataFrame):
                metric = metric.iloc[0]
            pairs.append(
                {
                    "activity_date": activity.activity_date,
                    "activity_hours": activity.activity_seconds / 3600,
                    "activity_types": activity.activity_types,
                    "sleep_date": sleep_date,
                    "hours_to_sleep": (
                        night["sleep_start"] - activity.last_activity_end
                    ).total_seconds()
                    / 3600,
                    "sleep_hours": float(night["sleep_minutes"]) / 60,
                    "sleep_score": (
                        float(night["legacy_garmin_score_10"]) * 10
                        if pd.notna(night["legacy_garmin_score_10"])
                        else float("nan")
                    ),
                    "overnight_hrv_display": (
                        metric["overnight_hrv_display"]
                        if metric is not None
                        else float("nan")
                    ),
                    "average_sleep_stress": (
                        metric["average_sleep_stress"]
                        if metric is not None
                        else float("nan")
                    ),
                }
            )
        pair_frame = pd.DataFrame(pairs)
        if pair_frame.empty:
            figure.add_annotation(
                text="No activity-to-subsequent-sleep pairs in the selected date range",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
            )
        else:
            predictor_column = (
                "hours_to_sleep" if predictor == "gap" else "activity_hours"
            )
            predictor_label = (
                "Time from activity end to sleep (hours)"
                if predictor == "gap"
                else "Activity duration (hours)"
            )
            customdata = pair_frame[
                [
                    "activity_date",
                    "activity_types",
                    "sleep_date",
                    "activity_hours",
                    "hours_to_sleep",
                ]
            ]
            for index, (title, column, unit, color) in enumerate(panels):
                row, col = divmod(index, 2)
                row += 1
                col += 1
                figure.add_trace(
                    go.Scatter(
                        name=title,
                        x=pair_frame[predictor_column],
                        y=pair_frame[column],
                        mode="markers",
                        marker=dict(color=color, size=7, opacity=0.78),
                        customdata=customdata,
                        showlegend=False,
                        hovertemplate=(
                            "Activity: %{customdata[1]}<br>"
                            "Activity date: %{customdata[0]}<br>"
                            "Following sleep: %{customdata[2]}<br>"
                            "Activity duration: %{customdata[3]:.2f} h<br>"
                            "Activity end → sleep: %{customdata[4]:.2f} h<br>"
                            f"{predictor_label}: %{{x:.2f}}<br>"
                            f"{title}: %{{y:.1f}} {unit}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=col,
                )
                valid = pair_frame[[predictor_column, column]].dropna().sort_values(
                    predictor_column
                )
                if len(valid) >= 2 and valid[predictor_column].nunique() > 1:
                    x_values = valid[predictor_column]
                    y_values = valid[column]
                    centered = x_values - x_values.mean()
                    slope = float(
                        (centered * (y_values - y_values.mean())).sum()
                        / (centered**2).sum()
                    )
                    trend_x = [float(x_values.min()), float(x_values.max())]
                    intercept = float(y_values.mean() - slope * x_values.mean())
                    figure.add_trace(
                        go.Scatter(
                            name="Trend",
                            x=trend_x,
                            y=[intercept + slope * value for value in trend_x],
                            mode="lines",
                            line=dict(color=color, width=1.5, dash="dash"),
                            hoverinfo="skip",
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )
                summary = _spearman_summary(
                    pair_frame[predictor_column], pair_frame[column]
                )
                if summary is not None:
                    correlation, count = summary
                    figure.layout.annotations[index].text = (
                        f"{title}<br><sup>Spearman ρ={correlation:+.2f} · n={count}</sup>"
                    )
                figure.update_xaxes(title_text=predictor_label, row=row, col=col)
                figure.update_yaxes(title_text=unit, row=row, col=col)
    figure.update_layout(
        title=(
            "Activity timing vs subsequent sleep"
            if predictor == "gap"
            else "Activity duration vs subsequent sleep"
        ),
        height=650,
        margin=dict(l=76, r=32, t=90, b=62),
    )
    return figure


def _activity_sleep_pairs(
    activities: pd.DataFrame, sessions: pd.DataFrame, metrics: pd.DataFrame
) -> pd.DataFrame:
    nights = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP") & (~sessions["zero_duration"])
    ].sort_values("sleep_start")
    if activities.empty or nights.empty:
        return pd.DataFrame()
    grouped = (
        activities.groupby("activity_date", as_index=False)
        .agg(
            activity_hours=("duration_seconds", lambda values: values.sum() / 3600),
            last_activity_end=("activity_end", "max"),
            activity_type=(
                "activity_type",
                lambda values: (
                    str(next(iter(set(values)))).replace("_", " ").title()
                    if len(set(values)) == 1
                    else "Mixed activities"
                ),
            ),
            calories=("calories", "sum"),
            average_hr=("average_hr", "mean"),
            max_hr=("max_hr", "max"),
            high_zone_seconds=(
                "hr_zone_4_seconds",
                lambda values: values.fillna(0).sum(),
            ),
            zone_5_seconds=(
                "hr_zone_5_seconds",
                lambda values: values.fillna(0).sum(),
            ),
        )
    )
    metric_index = metrics.set_index("date") if not metrics.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for activity in grouped.itertuples(index=False):
        candidates = nights[nights["sleep_start"] >= activity.last_activity_end]
        if candidates.empty:
            continue
        night = candidates.iloc[0]
        sleep_date = night["sleep_date"]
        metric = (
            metric_index.loc[sleep_date]
            if not metric_index.empty and sleep_date in metric_index.index
            else None
        )
        if isinstance(metric, pd.DataFrame):
            metric = metric.iloc[0]
        rows.append(
            {
                **activity._asdict(),
                "sleep_date": sleep_date,
                "sleep_hours": float(night["sleep_minutes"]) / 60,
                "sleep_score": (
                    float(night["legacy_garmin_score_10"]) * 10
                    if pd.notna(night["legacy_garmin_score_10"])
                    else float("nan")
                ),
                "overnight_hrv": (
                    metric.get("hrv_last_night_average")
                    if metric is not None
                    else float("nan")
                ),
                "sleep_stress": (
                    metric.get("average_sleep_stress")
                    if metric is not None
                    else float("nan")
                ),
                "high_zone_minutes": (
                    float(activity.high_zone_seconds or 0)
                    + float(activity.zone_5_seconds or 0)
                )
                / 60,
            }
        )
    return pd.DataFrame(rows)


def activity_type_comparison_figure(
    activities: pd.DataFrame, sessions: pd.DataFrame, metrics: pd.DataFrame
) -> go.Figure:
    pairs = _activity_sleep_pairs(activities, sessions, metrics)
    panels = (
        ("Following sleep duration", "sleep_hours", "hours"),
        ("Following Sleep score", "sleep_score", "score"),
        ("Following overnight HRV", "overnight_hrv", "ms"),
        ("Following sleep stress", "sleep_stress", ""),
    )
    figure = make_subplots(rows=2, cols=2, subplot_titles=[p[0] for p in panels])
    if pairs.empty:
        figure.add_annotation(text="No activity/sleep pairs", showarrow=False)
    else:
        palette = ["#087CFF", "#5B32D6", "#E8793F", "#4DB88A", "#EE55C0"]
        types = sorted(pairs["activity_type"].unique())
        for panel_index, (title, column, unit) in enumerate(panels):
            row, col = divmod(panel_index, 2)
            row += 1
            col += 1
            for type_index, activity_type in enumerate(types):
                values = pairs.loc[pairs["activity_type"] == activity_type, column].dropna()
                if values.empty:
                    continue
                figure.add_trace(
                    go.Box(
                        name=activity_type,
                        x=[activity_type] * len(values),
                        y=values,
                        boxmean=True,
                        boxpoints="all",
                        jitter=0.25,
                        marker_color=palette[type_index % len(palette)],
                        fillcolor="rgba(0,0,0,0)",
                        showlegend=panel_index == 0,
                        legendgroup=activity_type,
                        hovertemplate=f"{activity_type}<br>{title}: %{{y:.1f}} {unit}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )
            figure.update_yaxes(title_text=unit, row=row, col=col)
    figure.update_layout(
        title="Subsequent sleep by activity type",
        height=680,
        margin=dict(l=70, r=30, t=95, b=100),
        legend=dict(orientation="h", y=1.04),
        boxmode="group",
    )
    return figure


def activity_intensity_effect_figure(
    activities: pd.DataFrame, sessions: pd.DataFrame, metrics: pd.DataFrame
) -> go.Figure:
    pairs = _activity_sleep_pairs(activities, sessions, metrics)
    panels = (
        ("Calories", "calories", "kcal"),
        ("Average activity HR", "average_hr", "bpm"),
        ("Maximum activity HR", "max_hr", "bpm"),
        ("Time in HR zones 4–5", "high_zone_minutes", "min"),
    )
    figure = make_subplots(rows=2, cols=2, subplot_titles=[p[0] for p in panels])
    for index, (title, column, unit) in enumerate(panels):
        row, col = divmod(index, 2)
        row += 1
        col += 1
        if pairs.empty:
            continue
        valid = pairs[[column, "sleep_score"]].dropna()
        figure.add_trace(
            go.Scatter(
                x=valid[column],
                y=valid["sleep_score"],
                mode="markers",
                marker=dict(color="#4DB88A", size=7, opacity=0.8),
                showlegend=False,
                hovertemplate=f"{title}: %{{x:.1f}} {unit}<br>Following score: %{{y:.0f}}/100<extra></extra>",
            ),
            row=row,
            col=col,
        )
        summary = _spearman_summary(valid[column], valid["sleep_score"])
        if summary:
            rho, count = summary
            figure.layout.annotations[index].text = (
                f"{title}<br><sup>Spearman ρ={rho:+.2f} · n={count}</sup>"
            )
        figure.update_xaxes(title_text=unit, row=row, col=col)
        figure.update_yaxes(title_text="Following Sleep score", range=[0, 100], row=row, col=col)
    figure.update_layout(
        title="Activity intensity and heart-rate load",
        height=650,
        margin=dict(l=76, r=30, t=95, b=65),
    )
    return figure


def activity_to_sleep_gap_figure(
    activities: pd.DataFrame,
    sessions: pd.DataFrame,
    exclude_daytime_activities: bool = False,
) -> go.Figure:
    figure = go.Figure()
    nights = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (~sessions["zero_duration"])
    ].sort_values("sleep_start")
    rows: list[dict[str, object]] = []
    activity_frame = activities.copy()
    if exclude_daytime_activities and not activity_frame.empty:
        start_minutes = activity_frame["activity_start"].map(
            lambda value: value.hour * 60 + value.minute
        )
        end_minutes = activity_frame["activity_end"].map(
            lambda value: value.hour * 60 + value.minute
        )
        window_start, window_end = DAYTIME_WINDOW_MINUTES
        daytime = (start_minutes >= window_start) & (end_minutes <= window_end)
        activity_frame = activity_frame[~daytime]
    if not activity_frame.empty and not nights.empty:
        for activity in activity_frame.sort_values("activity_end").itertuples(index=False):
            following = nights[nights["sleep_start"] >= activity.activity_end]
            if following.empty:
                continue
            night = following.iloc[0]
            gap_hours = (
                night["sleep_start"] - activity.activity_end
            ).total_seconds() / 3600
            rows.append(
                {
                    "activity_end": activity.activity_end,
                    "activity_name": activity.activity_name,
                    "activity_type": str(activity.activity_type)
                    .replace("_", " ")
                    .title(),
                    "sleep_start": night["sleep_start"],
                    "gap_hours": gap_hours,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        figure.add_annotation(
            text="No activity-to-sleep pairs in the selected date range",
            showarrow=False,
        )
    else:
        palette = [
            "#FF7AB6",
            "#087CFF",
            "#5B32D6",
            "#2FD6A3",
            "#E8793F",
            "#E6B84F",
            "#EE55C0",
        ]
        for index, activity_type in enumerate(sorted(frame["activity_type"].unique())):
            subset = frame[frame["activity_type"] == activity_type]
            figure.add_trace(
                go.Scatter(
                    name=activity_type,
                    x=subset["activity_end"],
                    y=subset["gap_hours"],
                    mode="markers",
                    marker=dict(
                        color=palette[index % len(palette)], size=8, opacity=0.9
                    ),
                    customdata=subset[
                        ["activity_name", "activity_end", "sleep_start"]
                    ],
                    hovertemplate=(
                        "%{customdata[0]}<br>Ended: %{customdata[1]|%d/%m %H:%M}"
                        "<br>Sleep began: %{customdata[2]|%d/%m %H:%M}"
                        "<br>Gap: %{y:.1f} hours<extra></extra>"
                    ),
                )
            )
        median_gap = float(frame["gap_hours"].median())
        figure.add_hline(
            y=median_gap,
            line=dict(color="#E6B84F", width=1.5, dash="dot"),
            annotation_text=f"Median {median_gap:.1f} h",
            annotation_position="top left",
        )
    figure.update_layout(
        title=(
            "Time from activity ending to sleep · excluding "
            f"{DAYTIME_WINDOW_TEXT[0]}–{DAYTIME_WINDOW_TEXT[1]} activities"
            if exclude_daytime_activities
            else "Time from activity ending to sleep"
        ),
        height=470,
        margin=dict(l=76, r=34, t=75, b=62),
        xaxis=dict(title="Activity end date", type="date", showgrid=False),
        yaxis=dict(title="Hours until sleep begins", rangemode="tozero"),
        legend=dict(orientation="h", y=1.02),
    )
    return figure


def intensity_minutes_vs_subsequent_sleep_figure(
    metrics: pd.DataFrame, sessions: pd.DataFrame
) -> go.Figure:
    nights = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (~sessions["zero_duration"])
    ].sort_values("sleep_start")
    hrv_by_date = pd.Series(dtype=float)
    if not metrics.empty:
        hrv_values = metrics["hrv_last_night_average"].fillna(
            metrics["overnight_hrv"]
        )
        hrv_by_date = pd.Series(
            hrv_values.values, index=metrics["date"]
        ).groupby(level=0).first()
    stress_by_date = (
        metrics.groupby("date")["average_sleep_stress"].first()
        if not metrics.empty
        else pd.Series(dtype=float)
    )
    rows: list[dict[str, object]] = []
    if (
        not metrics.empty
        and "daily_intensity_minutes" in metrics
        and not nights.empty
    ):
        for day in metrics[
            metrics["daily_intensity_minutes"].notna()
        ].itertuples(index=False):
            cutoff = pd.Timestamp(day.date).tz_localize(TIMEZONE_NAME) + pd.Timedelta(
                hours=12
            )
            following = nights[nights["sleep_start"] >= cutoff]
            if following.empty:
                continue
            night = following.iloc[0]
            rows.append(
                {
                    "date": day.date,
                    "intensity_minutes": float(day.daily_intensity_minutes),
                    "sleep_date": night["sleep_date"],
                    "sleep_hours": float(night["sleep_minutes"]) / 60,
                    "sleep_score": (
                        float(night["legacy_garmin_score_10"]) * 10
                        if pd.notna(night["legacy_garmin_score_10"])
                        else float("nan")
                    ),
                    "overnight_hrv": hrv_by_date.get(
                        night["sleep_date"], float("nan")
                    ),
                    "sleep_stress": stress_by_date.get(
                        night["sleep_date"], float("nan")
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    figure = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=[
            "Following sleep duration",
            "Following Garmin Sleep score",
            "Following overnight HRV",
            "Following average sleep stress",
        ],
    )
    panels = (
        ("sleep_hours", "Hours asleep", None),
        ("sleep_score", "Garmin Sleep score", [0, 100]),
        ("overnight_hrv", "Overnight HRV (ms)", None),
        ("sleep_stress", "Average sleep stress", None),
    )
    for column_index, (outcome, y_title, y_range) in enumerate(panels, 1):
        if frame.empty:
            continue
        valid = frame[["intensity_minutes", outcome, "date", "sleep_date"]].dropna()
        figure.add_trace(
            go.Scatter(
                x=valid["intensity_minutes"],
                y=valid[outcome],
                mode="markers",
                marker=dict(color="#2FD6A3", size=8, opacity=0.82),
                customdata=valid[["date", "sleep_date"]],
                hovertemplate=(
                    "Activity day: %{customdata[0]}<br>"
                    "Intensity minutes: %{x:.0f}<br>"
                    "Following sleep date: %{customdata[1]}<br>"
                    f"{y_title}: %{{y:.1f}}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1,
            col=column_index,
        )
        summary = _spearman_summary(valid["intensity_minutes"], valid[outcome])
        if summary:
            rho, count = summary
            figure.layout.annotations[column_index - 1].text += (
                f"<br><sup>Spearman ρ={rho:+.2f} · n={count}</sup>"
            )
        figure.update_xaxes(
            title_text="Garmin intensity minutes that day",
            rangemode="tozero",
            row=1,
            col=column_index,
        )
        figure.update_yaxes(
            title_text=y_title,
            range=y_range,
            rangemode=None if y_range else "tozero",
            row=1,
            col=column_index,
        )
    if frame.empty:
        figure.add_annotation(
            text="No intensity-minute and subsequent-sleep pairs",
            showarrow=False,
        )
    figure.update_layout(
        title="Daily intensity minutes vs subsequent sleep",
        height=500,
        margin=dict(l=76, r=34, t=90, b=68),
    )
    return figure


def steps_floors_vs_subsequent_sleep_figure(
    metrics: pd.DataFrame, sessions: pd.DataFrame
) -> go.Figure:
    nights = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (~sessions["zero_duration"])
    ].sort_values("sleep_start")
    hrv_by_date = pd.Series(dtype=float)
    if not metrics.empty:
        hrv_values = metrics["hrv_last_night_average"].fillna(
            metrics["overnight_hrv"]
        )
        hrv_by_date = pd.Series(
            hrv_values.values, index=metrics["date"]
        ).groupby(level=0).first()
    stress_by_date = (
        metrics.groupby("date")["average_sleep_stress"].first()
        if not metrics.empty
        else pd.Series(dtype=float)
    )
    rows: list[dict[str, object]] = []
    required = {"daily_steps", "daily_floors_ascended"}
    if not metrics.empty and required.issubset(metrics.columns) and not nights.empty:
        usable_days = metrics[
            metrics[["daily_steps", "daily_floors_ascended"]]
            .notna()
            .any(axis=1)
        ]
        for day in usable_days.itertuples(index=False):
            cutoff = pd.Timestamp(day.date).tz_localize(TIMEZONE_NAME) + pd.Timedelta(
                hours=12
            )
            following = nights[nights["sleep_start"] >= cutoff]
            if following.empty:
                continue
            night = following.iloc[0]
            rows.append(
                {
                    "date": day.date,
                    "steps": day.daily_steps,
                    "floors": day.daily_floors_ascended,
                    "sleep_date": night["sleep_date"],
                    "sleep_hours": float(night["sleep_minutes"]) / 60,
                    "sleep_score": (
                        float(night["legacy_garmin_score_10"]) * 10
                        if pd.notna(night["legacy_garmin_score_10"])
                        else float("nan")
                    ),
                    "overnight_hrv": hrv_by_date.get(
                        night["sleep_date"], float("nan")
                    ),
                    "sleep_stress": stress_by_date.get(
                        night["sleep_date"], float("nan")
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    panels = (
        ("steps", "sleep_hours", "Daily steps", "Following sleep duration", "#087CFF"),
        ("steps", "sleep_score", "Daily steps", "Following Garmin Sleep score", "#087CFF"),
        ("steps", "overnight_hrv", "Daily steps", "Following overnight HRV", "#087CFF"),
        ("steps", "sleep_stress", "Daily steps", "Following average sleep stress", "#087CFF"),
        ("floors", "sleep_hours", "Floors climbed", "Following sleep duration", "#E8793F"),
        ("floors", "sleep_score", "Floors climbed", "Following Garmin Sleep score", "#E8793F"),
        ("floors", "overnight_hrv", "Floors climbed", "Following overnight HRV", "#E8793F"),
        ("floors", "sleep_stress", "Floors climbed", "Following average sleep stress", "#E8793F"),
    )
    figure = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[panel[3] for panel in panels],
        vertical_spacing=0.16,
    )
    for index, (predictor, outcome, x_title, y_title, color) in enumerate(panels):
        row, column = divmod(index, 4)
        row += 1
        column += 1
        if frame.empty:
            continue
        valid = frame[[predictor, outcome, "date", "sleep_date"]].dropna()
        figure.add_trace(
            go.Scatter(
                x=valid[predictor],
                y=valid[outcome],
                mode="markers",
                marker=dict(color=color, size=7, opacity=0.82),
                customdata=valid[["date", "sleep_date"]],
                hovertemplate=(
                    "Activity day: %{customdata[0]}<br>"
                    f"{x_title}: %{{x:.1f}}<br>"
                    "Following sleep date: %{customdata[1]}<br>"
                    f"{y_title}: %{{y:.1f}}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=column,
        )
        summary = _spearman_summary(valid[predictor], valid[outcome])
        if summary:
            rho, count = summary
            figure.layout.annotations[index].text += (
                f"<br><sup>{x_title} · Spearman ρ={rho:+.2f} · n={count}</sup>"
            )
        figure.update_xaxes(title_text=x_title, rangemode="tozero", row=row, col=column)
        figure.update_yaxes(
            title_text=(
                "Hours asleep"
                if outcome == "sleep_hours"
                else "Sleep score"
                if outcome == "sleep_score"
                else "Overnight HRV (ms)"
                if outcome == "overnight_hrv"
                else "Average sleep stress"
            ),
            range=([0, 100] if outcome == "sleep_score" else None),
            rangemode=(
                "tozero"
                if outcome in {"overnight_hrv", "sleep_stress"}
                else None
            ),
            row=row,
            col=column,
        )
    if frame.empty:
        figure.add_annotation(
            text="No steps/floors and subsequent-sleep pairs",
            showarrow=False,
        )
    figure.update_layout(
        title="Steps and floors climbed vs subsequent sleep",
        height=720,
        margin=dict(l=76, r=34, t=95, b=70),
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
