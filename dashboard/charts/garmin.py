"""Charts built only from Garmin measurements."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.config import TIMEZONE_NAME


def sleep_score_components_figure(metrics: pd.DataFrame) -> go.Figure:
    """Show Garmin sleep-score component qualifiers across time."""
    if metrics.empty:
        figure = go.Figure()
        figure.add_annotation(
            text="No Garmin sleep-score components in the selected date range",
            showarrow=False,
        )
        figure.update_layout(title="Garmin Sleep score components", height=360)
        return figure

    frame = metrics.sort_values("date").copy()
    qualifier_rank = {"POOR": 1, "FAIR": 2, "GOOD": 3, "EXCELLENT": 4}
    components = (
        ("Total duration", "score_duration_qualifier", None, ""),
        ("Sleep stress", "score_stress_qualifier", "average_sleep_stress", ""),
        (
            "Restlessness",
            "score_restlessness_qualifier",
            "restless_moment_count",
            " moments",
        ),
        ("Awake count", "score_awake_count_qualifier", "awake_count", ""),
        ("Deep sleep", "score_deep_qualifier", "score_deep_percentage", "%"),
        ("REM sleep", "score_rem_qualifier", "score_rem_percentage", "%"),
        ("Light sleep", "score_light_qualifier", "score_light_percentage", "%"),
    )
    z_rows: list[list[float | None]] = []
    text_rows: list[list[str]] = []
    for _, qualifier_column, value_column, suffix in components:
        qualifiers = (
            frame[qualifier_column]
            if qualifier_column in frame
            else pd.Series([None] * len(frame), index=frame.index)
        )
        values = (
            frame[value_column]
            if value_column is not None and value_column in frame
            else pd.Series([None] * len(frame), index=frame.index)
        )
        z_rows.append(
            [qualifier_rank.get(str(value).upper()) if pd.notna(value) else None for value in qualifiers]
        )
        row_text: list[str] = []
        for qualifier, value in zip(qualifiers, values):
            if pd.isna(qualifier):
                row_text.append("Unavailable")
                continue
            label = str(qualifier).replace("_", " ").title()
            if pd.notna(value):
                label += f" · {float(value):.0f}{suffix}"
            row_text.append(label)
        text_rows.append(row_text)

    colorscale = [
        [0.0, "#FF5A4D"],
        [0.2499, "#FF5A4D"],
        [0.25, "#FF9F3F"],
        [0.4999, "#FF9F3F"],
        [0.5, "#C5DC4A"],
        [0.7499, "#C5DC4A"],
        [0.75, "#3FD67A"],
        [1.0, "#3FD67A"],
    ]
    figure = go.Figure(
        go.Heatmap(
            x=pd.to_datetime(frame["date"]),
            y=[component[0] for component in components],
            z=z_rows,
            text=text_rows,
            zmin=0.5,
            zmax=4.5,
            colorscale=colorscale,
            colorbar=dict(
                title="Qualifier",
                tickmode="array",
                tickvals=[1, 2, 3, 4],
                ticktext=["Poor", "Fair", "Good", "Excellent"],
            ),
            hovertemplate=(
                "%{x|%d/%m/%Y}<br><b>%{y}</b><br>%{text}<extra></extra>"
            ),
            xgap=1,
            ygap=1,
        )
    )
    figure.update_layout(
        title="Garmin Sleep score components",
        height=470,
        margin=dict(l=125, r=80, t=70, b=60),
        xaxis=dict(title="Sleep date", tickformat="%d/%m", showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False),
    )
    return figure


def overnight_recovery_figure(metrics: pd.DataFrame) -> go.Figure:
    """Show key overnight recovery measurements as aligned small multiples."""
    panels = (
        ("Overnight HRV", "hrv_last_night_average", "ms", "#4DB88A"),
        ("Sleeping heart rate", "average_sleep_hr", "bpm", "#E8793F"),
        ("Sleep stress", "average_sleep_stress", "", "#E6B84F"),
        ("Body Battery gained", "body_battery_change", "", "#2FD6A3"),
        ("Sleeping respiration", "average_respiration", "brpm", "#087CFF"),
        ("Average SpO₂", "average_spo2", "%", "#EE55C0"),
    )
    figure = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
        subplot_titles=[panel[0] for panel in panels],
    )
    if metrics.empty:
        figure.add_annotation(
            text="No overnight recovery measurements in the selected date range",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
    else:
        frame = metrics.sort_values("date").copy()
        frame["overnight_hrv_display"] = frame[
            "hrv_last_night_average"
        ].fillna(frame["overnight_hrv"])
        for index, (title, column, unit, color) in enumerate(panels):
            row, col = divmod(index, 2)
            row += 1
            col += 1
            values = (
                frame["overnight_hrv_display"]
                if column == "hrv_last_night_average"
                else frame[column]
            )
            figure.add_trace(
                go.Scatter(
                    name=title,
                    x=pd.to_datetime(frame["date"]),
                    y=values,
                    mode="lines",
                    connectgaps=False,
                    line=dict(color=color, width=1.4),
                    showlegend=False,
                    hovertemplate=(
                        f"%{{x|%d/%m/%Y}}<br>{title}: %{{y:.1f}} {unit}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            figure.update_yaxes(title_text=unit, showgrid=True, row=row, col=col)
    figure.update_xaxes(tickformat="%d/%m", showgrid=False)
    figure.update_layout(
        title="Overnight recovery panel",
        height=760,
        margin=dict(l=72, r=32, t=90, b=55),
        hovermode="x unified",
    )
    return figure


def daily_physiological_load_figure(metrics: pd.DataFrame) -> go.Figure:
    """Show daily activity and physiological load as aligned small multiples."""
    panels = (
        ("Steps", "daily_steps", "steps", "#087CFF"),
        ("Intensity minutes", "daily_intensity_minutes", "min", "#5B32D6"),
        ("Resting heart rate", "daily_resting_heart_rate", "bpm", "#E8793F"),
        ("Average stress", "daily_average_stress", "", "#E6B84F"),
        ("Body Battery drained", "daily_body_battery_drained", "", "#EE55C0"),
        ("Active calories", "daily_active_calories", "kcal", "#4DB88A"),
    )
    figure = make_subplots(
        rows=3,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
        subplot_titles=[panel[0] for panel in panels],
    )
    if metrics.empty:
        figure.add_annotation(
            text="No daily physiological-load data in the selected date range",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
    else:
        frame = metrics.sort_values("date")
        for index, (title, column, unit, color) in enumerate(panels):
            row, col = divmod(index, 2)
            row += 1
            col += 1
            values = (
                frame[column]
                if column in frame
                else pd.Series([float("nan")] * len(frame), index=frame.index)
            )
            figure.add_trace(
                go.Scatter(
                    name=title,
                    x=pd.to_datetime(frame["date"]),
                    y=values,
                    mode="lines",
                    connectgaps=False,
                    line=dict(color=color, width=1.4),
                    showlegend=False,
                    hovertemplate=(
                        f"%{{x|%d/%m/%Y}}<br>{title}: %{{y:.1f}} {unit}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            figure.update_yaxes(title_text=unit, rangemode="tozero", row=row, col=col)
    figure.update_xaxes(tickformat="%d/%m", showgrid=False)
    figure.update_layout(
        title="Daily physiological load",
        height=760,
        margin=dict(l=72, r=32, t=90, b=55),
        hovermode="x unified",
    )
    return figure


def awakenings_efficiency_figure(metrics: pd.DataFrame) -> go.Figure:
    """Show Garmin awakening measures and stage-derived sleep efficiency."""
    panels = (
        ("Awake time", "awake_minutes", "min", "#E14E5C"),
        ("Awake count", "awake_count", "count", "#FFB547"),
        ("Sleep efficiency", "sleep_efficiency", "%", "#4DB88A"),
        ("Restlessness rating", "restlessness_rating", "rating", "#5B32D6"),
    )
    figure = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.17,
        horizontal_spacing=0.10,
        subplot_titles=[panel[0] for panel in panels],
    )
    if metrics.empty:
        figure.add_annotation(
            text="No Garmin awakening data in the selected date range",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
    else:
        frame = metrics.sort_values("date").copy()
        asleep_minutes = frame[
            ["deep_sleep_minutes", "light_sleep_minutes", "rem_sleep_minutes"]
        ].fillna(0).sum(axis=1)
        measured_minutes = asleep_minutes + frame["awake_minutes"].fillna(0)
        frame["sleep_efficiency"] = (
            asleep_minutes / measured_minutes.where(measured_minutes > 0) * 100
        )
        rating_values = {"POOR": 1, "FAIR": 2, "GOOD": 3, "EXCELLENT": 4}
        frame["restlessness_rating"] = frame[
            "score_restlessness_qualifier"
        ].map(rating_values)
        for index, (title, column, unit, color) in enumerate(panels):
            row, col = divmod(index, 2)
            row += 1
            col += 1
            figure.add_trace(
                go.Scatter(
                    name=title,
                    x=pd.to_datetime(frame["date"]),
                    y=frame[column],
                    mode="lines",
                    connectgaps=False,
                    line=dict(color=color, width=1.4),
                    showlegend=False,
                    hovertemplate=(
                        f"%{{x|%d/%m/%Y}}<br>{title}: %{{y:.1f}} {unit}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            if column == "restlessness_rating":
                figure.update_yaxes(
                    tickmode="array",
                    tickvals=[1, 2, 3, 4],
                    ticktext=["Poor", "Fair", "Good", "Excellent"],
                    range=[0.5, 4.5],
                    row=row,
                    col=col,
                )
            elif column == "sleep_efficiency":
                figure.update_yaxes(title_text=unit, range=[0, 100], row=row, col=col)
            else:
                figure.update_yaxes(title_text=unit, rangemode="tozero", row=row, col=col)
    figure.update_xaxes(tickformat="%d/%m", showgrid=False)
    figure.update_layout(
        title="Awakenings and sleep efficiency",
        height=610,
        margin=dict(l=74, r=32, t=90, b=55),
        hovermode="x unified",
    )
    return figure


def intraday_recovery_timeline_figure(
    intraday: pd.DataFrame,
    activities: pd.DataFrame,
    sessions: pd.DataFrame,
    selected_date: date,
) -> go.Figure:
    frame = intraday[intraday["date"] == selected_date].sort_values("timestamp")
    figure = go.Figure()
    for name, column, color in (
        ("Heart rate", "heart_rate", "#E8793F"),
        ("Stress", "stress", "#E6B84F"),
        ("Body Battery", "body_battery", "#4DB88A"),
    ):
        samples = frame.loc[frame[column].notna(), ["timestamp", column]]
        figure.add_trace(
            go.Scatter(
                name=name,
                x=samples["timestamp"],
                y=samples[column],
                mode="lines",
                connectgaps=False,
                line=dict(color=color, width=1),
            )
        )
    for activity in activities[activities["activity_date"] == selected_date].itertuples(index=False):
        figure.add_vrect(x0=activity.activity_start, x1=activity.activity_end, fillcolor="#2FD6A3", opacity=0.18, line_width=0, annotation_text=activity.activity_name)
    day_start = pd.Timestamp(selected_date).tz_localize(TIMEZONE_NAME)
    day_end = day_start + pd.Timedelta(days=1)
    for night in sessions[(sessions["wake_time"] > day_start) & (sessions["sleep_start"] < day_end)].itertuples(index=False):
        figure.add_vrect(x0=max(night.sleep_start, day_start.to_pydatetime()), x1=min(night.wake_time, day_end.to_pydatetime()), fillcolor="#5B32D6", opacity=0.12, line_width=0)
    figure.update_layout(
        title=f"Intraday recovery · {selected_date:%d/%m/%Y}",
        height=500,
        yaxis=dict(title="Value", rangemode="tozero"),
        xaxis=dict(title="Time", tickformat="%H:%M", range=[day_start, day_end]),
        legend=dict(orientation="h", y=1.02),
        hovermode="x unified",
    )
    return figure


def heart_rate_24_hour_heatmap(
    intraday: pd.DataFrame,
    activities: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> go.Figure:
    frame = intraday[intraday["heart_rate"].notna()].copy()
    figure = go.Figure()
    if not frame.empty:
        frame["minute_of_day"] = (
            frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
        )
        frame["minute_bin"] = (frame["minute_of_day"] // 2) * 2
        matrix = frame.pivot_table(
            index="minute_bin",
            columns="date",
            values="heart_rate",
            aggfunc="mean",
        ).reindex(index=range(0, 1440, 2))
        all_dates = pd.date_range(start_date, end_date, freq="D")
        matrix = matrix.reindex(columns=[value.date() for value in all_dates])
        observed = frame["heart_rate"].dropna()
        color_min = max(35.0, float(observed.quantile(0.01)))
        color_max = max(color_min + 1, float(observed.quantile(0.995)))
        figure.add_trace(
            go.Heatmap(
                x=all_dates,
                y=matrix.index / 60,
                z=matrix.to_numpy(),
                zmin=color_min,
                zmax=color_max,
                colorscale=[
                    [0.0, "#172B55"],
                    [0.25, "#087CFF"],
                    [0.5, "#2FD6A3"],
                    [0.7, "#E6B84F"],
                    [0.85, "#E8793F"],
                    [1.0, "#FF5A4D"],
                ],
                colorbar=dict(title="bpm", thickness=14),
                hovertemplate=(
                    "%{x|%d/%m/%Y}<br>%{y:.2f} hours<br>"
                    "Heart rate: %{z:.0f} bpm<extra></extra>"
                ),
                xgap=0.2,
                ygap=0,
                connectgaps=False,
            )
        )
        if not activities.empty:
            activity_frame = activities.copy()
            activity_frame["start_hour"] = activity_frame["activity_start"].map(
                lambda value: value.hour
                + value.minute / 60
                + value.second / 3600
            )
            activity_frame["duration_hours"] = (
                activity_frame["duration_seconds"] / 3600
            )
            activity_frame["plot_date"] = pd.to_datetime(
                activity_frame["activity_date"]
            ) + pd.Timedelta(hours=12)
            figure.add_trace(
                go.Bar(
                    name="Recorded activity",
                    x=activity_frame["plot_date"],
                    y=activity_frame["duration_hours"],
                    base=activity_frame["start_hour"],
                    width=0.72 * 24 * 60 * 60 * 1000,
                    marker=dict(color="#FF7AB6", opacity=1.0, line_width=0),
                    customdata=activity_frame[
                        ["activity_name", "activity_start", "activity_end"]
                    ],
                    hovertemplate=(
                        "%{customdata[0]}<br>"
                        "%{customdata[1]|%d/%m %H:%M}–"
                        "%{customdata[2]|%H:%M}<extra></extra>"
                    ),
                )
            )
    else:
        figure.add_annotation(
            text="No heart-rate samples in the selected date range",
            showarrow=False,
        )
    figure.update_layout(
        title="Heart rate across 24 hours",
        height=720,
        margin=dict(l=78, r=50, t=70, b=65),
        xaxis=dict(
            title="Date",
            type="date",
            range=[pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(days=1)],
            showgrid=False,
            minallowed=pd.Timestamp(start_date),
            maxallowed=pd.Timestamp(end_date) + pd.Timedelta(days=1),
        ),
        yaxis=dict(
            title="Time of day",
            range=[0, 24],
            tickmode="array",
            tickvals=[0, 4, 8, 12, 16, 20, 24],
            ticktext=["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "24:00"],
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        ),
        barmode="overlay",
        legend=dict(orientation="h", y=1.02),
    )
    return figure


def daily_calories_stacked_figure(metrics: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    required = {"daily_total_calories", "daily_active_calories"}
    if metrics.empty or not required.issubset(metrics.columns):
        figure.add_annotation(
            text="No daily calorie data in the selected date range",
            showarrow=False,
        )
    else:
        frame = metrics[
            ["date", "daily_total_calories", "daily_active_calories"]
        ].copy()
        frame = frame[frame["daily_total_calories"].notna()].sort_values("date")
        frame["active_calories"] = frame["daily_active_calories"].fillna(0).clip(
            lower=0
        )
        frame["resting_calories"] = (
            frame["daily_total_calories"] - frame["active_calories"]
        ).clip(lower=0)
        figure.add_trace(
            go.Bar(
                name="Resting calories",
                x=frame["date"],
                y=frame["resting_calories"],
                marker=dict(color="#5B32D6", line_width=0),
                hovertemplate="%{x|%d/%m/%Y}<br>Resting: %{y:.0f} kcal<extra></extra>",
            )
        )
        figure.add_trace(
            go.Bar(
                name="Active calories",
                x=frame["date"],
                y=frame["active_calories"],
                marker=dict(color="#FF7AB6", line_width=0),
                hovertemplate="%{x|%d/%m/%Y}<br>Active: %{y:.0f} kcal<extra></extra>",
            )
        )
    figure.update_layout(
        title="Daily calories burned",
        height=470,
        barmode="stack",
        bargap=0.2,
        margin=dict(l=76, r=34, t=70, b=62),
        xaxis=dict(title="Date", type="date", showgrid=False),
        yaxis=dict(title="Calories burned (kcal)", rangemode="tozero"),
        legend=dict(orientation="h", y=1.02),
        hovermode="x unified",
    )
    return figure
