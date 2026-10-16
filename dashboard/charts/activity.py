"""Charts on the Activity & sleep tab: activity calendars and the following night."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from dashboard.charts.common import SCORE_COLOR, WEEKDAY_ORDER
from dashboard.charts.sleep_metrics import (
    SLEEP_METRICS,
    _add_correlation,
    _add_panel_header,
    _add_trend,
    _no_pairs,
    _plot_sleep_metric_row,
    _sleep_metric_grid,
)
from dashboard.config import DAYTIME_WINDOW_MINUTES, DAYTIME_WINDOW_TEXT, TIMEZONE_NAME


def activity_effect_on_sleep_figure(
    activities: pd.DataFrame,
    sessions: pd.DataFrame,
    metrics: pd.DataFrame,
    predictor: str = "duration",
) -> go.Figure:
    """Relate each activity day's load to its next recorded night sleep."""
    figure = _sleep_metric_grid()
    figure.update_layout(
        title=(
            "Activity timing vs the following night"
            if predictor == "gap"
            else "Activity duration vs the following night"
        )
    )
    nights = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (~sessions["zero_duration"])
    ].sort_values("sleep_start")
    if activities.empty or nights.empty:
        return _no_pairs(figure, "No activity and following-night pairs in the selected date range")
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
                "activity_date": f"{activity.activity_date:%d.%m.%Y}",
                "activity_hours": activity.activity_seconds / 3600,
                "activity_types": activity.activity_types,
                "sleep_date": f"{sleep_date:%d.%m.%Y}",
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
                "overnight_hrv": (
                    metric["overnight_hrv_display"] if metric is not None else float("nan")
                ),
                "sleep_stress": (
                    metric["average_sleep_stress"] if metric is not None else float("nan")
                ),
            }
        )
    pair_frame = pd.DataFrame(pairs)
    if pair_frame.empty:
        return _no_pairs(figure, "No activity and following-night pairs in the selected date range")
    _plot_sleep_metric_row(
        figure,
        pair_frame,
        "hours_to_sleep" if predictor == "gap" else "activity_hours",
        "Hours from activity end to sleep" if predictor == "gap" else "Activity duration (hours)",
        row=1,
        hover_columns=["activity_types", "activity_date", "sleep_date"],
        hover_prefix="%{customdata[0]} on %{customdata[1]}<br>Following night: %{customdata[2]}<br>",
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
    figure = _sleep_metric_grid(height=470)
    figure.update_layout(
        title="The following night by activity type",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0),
        margin=dict(l=70, r=30, t=100, b=60),
        boxmode="group",
    )
    if pairs.empty:
        return _no_pairs(figure, "No activity and following-night pairs")
    palette = ["#087CFF", "#5B32D6", "#E8793F", "#4DB88A", "#EE55C0"]
    types = sorted(pairs["activity_type"].unique())
    for col, (column, name, _, unit, _, _) in enumerate(SLEEP_METRICS, 1):
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
                    showlegend=col == 1,
                    legendgroup=activity_type,
                    hovertemplate=f"{activity_type}<br>Following {name.lower()}: %{{y:.1f}} {unit}<extra></extra>",
                ),
                row=1,
                col=col,
            )
        # The legend names the types, so the category labels would only repeat it.
        figure.update_xaxes(showticklabels=False, row=1, col=col)
    return figure


def activity_intensity_effect_figure(
    activities: pd.DataFrame, sessions: pd.DataFrame, metrics: pd.DataFrame
) -> go.Figure:
    """Four activity-load measures against the following night's sleep score."""
    pairs = _activity_sleep_pairs(activities, sessions, metrics)
    panels = (
        ("Calories (kcal)", "calories"),
        ("Average activity HR (bpm)", "average_hr"),
        ("Maximum activity HR (bpm)", "max_hr"),
        ("Time in HR zones 4–5 (min)", "high_zone_minutes"),
    )
    # Same one-row layout as the sleep-metric charts; every panel is sleep score.
    figure = make_subplots(rows=1, cols=len(panels), horizontal_spacing=0.055)
    figure.update_layout(
        title="Activity intensity vs the following sleep score",
        height=430,
        margin=dict(l=50, r=30, t=80, b=80),
    )
    figure.update_yaxes(range=[0, 100])
    for col in range(1, len(panels) + 1):
        _add_panel_header(figure, 1, col, "Sleep score", "0–100", SCORE_COLOR)
    if pairs.empty:
        return _no_pairs(figure, "No activity and following-night pairs")
    for col, (label, column) in enumerate(panels, 1):
        valid = pairs[[column, "sleep_score"]].dropna()
        figure.add_trace(
            go.Scatter(
                x=valid[column],
                y=valid["sleep_score"],
                mode="markers",
                marker=dict(color=SCORE_COLOR, size=7, opacity=0.8),
                showlegend=False,
                hovertemplate=f"{label}: %{{x:.1f}}<br>Following sleep score: %{{y:.0f}}/100<extra></extra>",
            ),
            row=1,
            col=col,
        )
        _add_trend(figure, 1, col, valid[column], valid["sleep_score"], SCORE_COLOR)
        _add_correlation(figure, 1, col, valid[column], valid["sleep_score"])
        figure.update_xaxes(title_text=label, row=1, col=col)
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
                        "%{customdata[0]}<br>Ended: %{customdata[1]|%d.%m %H:%M}"
                        "<br>Sleep began: %{customdata[2]|%d.%m %H:%M}"
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
        xaxis=dict(title="Activity end date", type="date", tickformat="%d.%m", showgrid=False),
        yaxis=dict(title="Hours until sleep begins", rangemode="tozero"),
        legend=dict(orientation="h", y=1.02),
    )
    return figure


def _daily_load_pairs(metrics: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Each day's Garmin load paired with the first night starting after its noon."""
    nights = sessions[
        (sessions["session_type"] == "NIGHT_SLEEP")
        & (~sessions["zero_duration"])
    ].sort_values("sleep_start")
    if metrics.empty or nights.empty:
        return pd.DataFrame()
    by_date = metrics.groupby("date").first()
    hrv_by_date = by_date["hrv_last_night_average"].fillna(by_date["overnight_hrv"])
    stress_by_date = by_date["average_sleep_stress"]
    rows: list[dict[str, object]] = []
    for day, values in by_date.iterrows():
        cutoff = pd.Timestamp(day).tz_localize(TIMEZONE_NAME) + pd.Timedelta(hours=12)
        following = nights[nights["sleep_start"] >= cutoff]
        if following.empty:
            continue
        night = following.iloc[0]
        rows.append(
            {
                "date": f"{day:%d.%m.%Y}",
                "intensity_minutes": values.get("daily_intensity_minutes"),
                "steps": values.get("daily_steps"),
                "floors": values.get("daily_floors_ascended"),
                "sleep_date": f"{night['sleep_date']:%d.%m.%Y}",
                "sleep_hours": float(night["sleep_minutes"]) / 60,
                "sleep_score": (
                    float(night["legacy_garmin_score_10"]) * 10
                    if pd.notna(night["legacy_garmin_score_10"])
                    else float("nan")
                ),
                "overnight_hrv": hrv_by_date.get(night["sleep_date"], float("nan")),
                "sleep_stress": stress_by_date.get(night["sleep_date"], float("nan")),
            }
        )
    return pd.DataFrame(rows)


DAILY_LOAD_HOVER = "Day: %{customdata[0]}<br>Following night: %{customdata[1]}<br>"


def intensity_minutes_vs_subsequent_sleep_figure(
    metrics: pd.DataFrame, sessions: pd.DataFrame
) -> go.Figure:
    frame = _daily_load_pairs(metrics, sessions)
    figure = _sleep_metric_grid()
    figure.update_layout(title="Daily intensity minutes vs the following night")
    if frame.empty or frame["intensity_minutes"].isna().all():
        return _no_pairs(figure, "No intensity-minute and following-night pairs")
    _plot_sleep_metric_row(
        figure, frame, "intensity_minutes", "Garmin intensity minutes that day", 1,
        ["date", "sleep_date"], DAILY_LOAD_HOVER,
    )
    figure.update_xaxes(rangemode="tozero")
    return figure


def steps_floors_vs_subsequent_sleep_figure(
    metrics: pd.DataFrame, sessions: pd.DataFrame, predictor: str = "steps"
) -> go.Figure:
    """Daily steps (predictor="steps") or floors climbed ("floors") vs the following night."""
    label = "Daily steps" if predictor == "steps" else "Floors climbed"
    frame = _daily_load_pairs(metrics, sessions)
    figure = _sleep_metric_grid()
    figure.update_layout(title=f"{label} vs the following night")
    if frame.empty or frame[predictor].isna().all():
        return _no_pairs(figure, f"No {label.lower()} and following-night pairs")
    _plot_sleep_metric_row(
        figure, frame, predictor, label, 1, ["date", "sleep_date"], DAILY_LOAD_HOVER
    )
    figure.update_xaxes(rangemode="tozero")
    return figure


# Calendar cells: days with nothing recorded get a box slightly lighter than
# the page; recorded days run from dark to bright green with the minutes.
CALENDAR_EMPTY_COLOR = "#1C2029"
CALENDAR_COLORSCALE = [[0.0, "#1E3B30"], [0.5, "#257A59"], [1.0, "#2FD6A3"]]
# The calendar always shows this many weeks, so cells keep one size: shorter
# ranges leave blank columns, longer ones pan back through earlier weeks.
CALENDAR_VISIBLE_WEEKS = 27  # 180 days can touch 27 calendar weeks
# Cell inset and corner radius, in cell widths.
CALENDAR_CELL_INSET = 0.07
CALENDAR_CELL_RADIUS = 0.14


def _rounded_cell(x: float, y: float) -> str:
    """SVG path of a rounded square centred on (x, y) in axis units."""
    x0, x1 = x - 0.5 + CALENDAR_CELL_INSET, x + 0.5 - CALENDAR_CELL_INSET
    y0, y1 = y - 0.5 + CALENDAR_CELL_INSET, y + 0.5 - CALENDAR_CELL_INSET
    r = CALENDAR_CELL_RADIUS
    return (
        f"M {x0 + r},{y0} L {x1 - r},{y0} Q {x1},{y0} {x1},{y0 + r} "
        f"L {x1},{y1 - r} Q {x1},{y1} {x1 - r},{y1} L {x0 + r},{y1} "
        f"Q {x0},{y1} {x0},{y1 - r} L {x0},{y0 + r} Q {x0},{y0} {x0 + r},{y0} Z"
    )


def _calendar_heatmap(
    daily: pd.DataFrame,
    minutes: pd.Series,
    hover_text: pd.Series,
    title: str,
    empty_text: str,
) -> go.Figure:
    """Week-by-weekday grid of rounded square cells for every day in the range.

    ``minutes`` and ``hover_text`` are indexed by date; days missing from them
    show as empty boxes. Cells are drawn as shapes (heatmap cells can't have
    rounded corners); an invisible scatter on top carries the hover text.
    """
    dates = pd.to_datetime(pd.Series(sorted(daily["date"].unique())))
    figure = go.Figure()
    figure.update_layout(
        title=title,
        height=330,
        margin=dict(l=80, r=65, t=64, b=56),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Month", showgrid=False, zeroline=False),
        yaxis=dict(
            title="Day of week",
            tickmode="array",
            tickvals=list(range(7)),
            ticktext=WEEKDAY_ORDER,
            range=[6.5, -0.5],
            showgrid=False,
            zeroline=False,
            fixedrange=True,
            # Square cells: one week is as wide as one day is tall.
            scaleanchor="x",
            scaleratio=1,
        ),
    )
    if dates.empty:
        return figure
    week_starts = dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")
    weeks = sorted(week_starts.drop_duplicates())
    week_index = {week: index for index, week in enumerate(weeks)}
    observed = minutes.dropna()
    top = max(1.0, float(observed.max())) if not observed.empty else 1.0
    colors = sample_colorscale(
        CALENDAR_COLORSCALE,
        [min(float(observed.get(day, 0)) / top, 1.0) for day in dates.dt.date],
    )
    xs, ys, hovers = [], [], []
    for timestamp, week_start, color in zip(dates, week_starts, colors):
        x, y = week_index[week_start], timestamp.dayofweek
        day = timestamp.date()
        recorded = day in observed.index
        figure.add_shape(
            type="path",
            path=_rounded_cell(x, y),
            fillcolor=color if recorded else CALENDAR_EMPTY_COLOR,
            line=dict(width=0),
            layer="below",
        )
        xs.append(x)
        ys.append(y)
        hovers.append(
            f"{timestamp:%d.%m.%Y}<br>{hover_text[day] if recorded else empty_text}"
        )
    figure.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            # Invisible hover targets; the marker colours only feed the colour bar.
            marker=dict(
                size=14,
                opacity=0,
                color=[float(observed.get(day, 0)) for day in dates.dt.date],
                cmin=0,
                cmax=top,
                colorscale=CALENDAR_COLORSCALE,
                showscale=True,
                colorbar=dict(title="Minutes", thickness=12),
            ),
            customdata=hovers,
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        )
    )
    last = len(weeks) - 1
    first_visible = last - CALENDAR_VISIBLE_WEEKS + 1
    # One label per month, on the first week that starts in it; the year is
    # added in January and on the first label.
    tick_positions, tick_labels = [], []
    for index, week in enumerate(weeks):
        week = pd.Timestamp(week)
        if index and week.month == pd.Timestamp(weeks[index - 1]).month:
            continue
        tick_positions.append(index)
        tick_labels.append(f"{week:%b %Y}" if not tick_labels or week.month == 1 else f"{week:%b}")
    figure.update_xaxes(
        range=[first_visible - 0.5, last + 0.5],
        minallowed=min(0, first_visible) - 0.5,
        maxallowed=last + 0.5,
        tickmode="array",
        tickvals=tick_positions,
        ticktext=tick_labels,
    )
    if len(weeks) > CALENDAR_VISIBLE_WEEKS:
        figure.update_layout(meta=dict(pan=True, pan_group="calendar"))
    return figure


def calendar_activity_heatmap(
    daily: pd.DataFrame, activities: pd.DataFrame
) -> go.Figure:
    if activities.empty:
        minutes = pd.Series(dtype=float)
        hover = pd.Series(dtype=object)
    else:
        grouped = activities.groupby("activity_date").agg(
            names=("activity_name", lambda values: ", ".join(dict.fromkeys(map(str, values)))),
            minutes=("duration_seconds", lambda values: values.sum() / 60),
        )
        minutes = grouped["minutes"]
        hover = grouped["names"] + "<br>" + grouped["minutes"].round().astype(int).astype(str) + " minutes"
    return _calendar_heatmap(
        daily, minutes, hover, "Recorded activity calendar", "No recorded activity"
    )


def calendar_intensity_minutes_heatmap(
    daily: pd.DataFrame, metrics: pd.DataFrame
) -> go.Figure:
    minutes = (
        metrics.groupby("date")["daily_intensity_minutes"].first().dropna()
        if not metrics.empty and "daily_intensity_minutes" in metrics
        else pd.Series(dtype=float)
    )
    hover = "Intensity minutes: " + minutes.round().astype(int).astype(str)
    return _calendar_heatmap(
        daily, minutes, hover, "Intensity minutes calendar", "No intensity minutes recorded"
    )
