from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent
COLLECTORS = (
    "sleep",
    "hrv",
    "daily_health",
    "body",
    "activities",
    "training",
    "menstrual",
)
PROFILE_COLLECTORS = {
    "daily": ("sleep", "hrv", "daily_health", "activities"),
    "weekly": ("body", "training", "menstrual"),
    "backfill": COLLECTORS,
}
PROFILE_BUDGETS = {"daily": 50, "weekly": 20, "backfill": 100}
# Catching up after missed runs (see --catch-up): extra requests allowed per
# missed day, and at most this many extra per run; the rest follows next run.
CATCH_UP_REQUESTS_PER_DAY = 10
CATCH_UP_EXTRA_REQUESTS = 90
STATE_FILE = "sync_state.json"


def _default_private_root() -> Path:
    """Write to the dashboard's data folder, never into the sample data.

    Uses ``GARMINXSLEEP_DATA_DIR`` when set (the folder the dashboard reads),
    otherwise the Git-ignored ``private/`` folder in the project.
    """
    configured = os.getenv("GARMINXSLEEP_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    if SCRIPT_DIRECTORY.name == "dashboard" and (PROJECT_ROOT / "pyproject.toml").exists():
        return PROJECT_ROOT / "private"
    return SCRIPT_DIRECTORY / "garmin-data"


def _load_client(
    client_path: Path | None,
) -> tuple[type[Any], tuple[type[Exception], ...]]:
    """Load a specified, adjacent, project-vendored, or installed client."""
    configured_path = os.getenv("GARMINCONNECT_PATH")
    candidates = [
        client_path,
        Path(configured_path).expanduser() if configured_path else None,
        SCRIPT_DIRECTORY / "python-garminconnect",
        SCRIPT_DIRECTORY.parent / "python-garminconnect",
    ]
    selected = next(
        (
            candidate.resolve()
            for candidate in candidates
            if candidate is not None
            and (candidate / "garminconnect" / "__init__.py").is_file()
        ),
        None,
    )
    if selected is not None:
        sys.path.insert(0, str(selected))

    try:
        from garminconnect import Garmin
        from garminconnect.exceptions import (
            GarminConnectAuthenticationError,
            GarminConnectConnectionError,
            GarminConnectTooManyRequestsError,
        )
    except ImportError as error:
        raise RuntimeError(
            "Unable to find python-garminconnect. Put its repository beside "
            "garmin_sync.py, pass --client-path, set GARMINCONNECT_PATH, or "
            "install garminconnect in this Python environment."
        ) from error

    return Garmin, (
        GarminConnectAuthenticationError,
        GarminConnectConnectionError,
        GarminConnectTooManyRequestsError,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _login(private_root: Path, client_path: Path | None) -> Any:
    Garmin, client_errors = _load_client(client_path)
    tokenstore = private_root / "tokens"
    tokenstore.mkdir(parents=True, exist_ok=True, mode=0o700)

    try:
        client = Garmin()
        client.login(str(tokenstore))
        print("Authenticated using the saved Garmin token.")
        return client
    except client_errors:
        print("No usable saved Garmin token. Interactive sign-in is required.")

    email = os.getenv("GARMIN_EMAIL") or input("Garmin email: ").strip()
    password = os.getenv("GARMIN_PASSWORD") or getpass.getpass("Garmin password: ")
    client = Garmin(
        email=email,
        password=password,
        prompt_mfa=lambda: input("Garmin MFA code: ").strip(),
    )
    client.login(str(tokenstore))
    print(f"Authentication succeeded. Token saved under {tokenstore}.")
    return client


class RequestBudgetReached(Exception):
    """Raised before a request would exceed the configured hard limit."""


class GarminRateLimited(Exception):
    """Raised to stop the entire run immediately after a Garmin 429."""


class Collector:
    def __init__(
        self,
        raw_root: Path,
        max_requests: int,
        request_delay: float,
        refresh_existing: bool,
    ) -> None:
        self.raw_root = raw_root
        self.max_requests = max_requests
        self.request_delay = request_delay
        self.refresh_existing = refresh_existing
        self.requests_used = 0
        self.saved_files: list[str] = []
        self.skipped_files: list[str] = []
        self.failures: list[dict[str, str]] = []
        self._last_request_at: float | None = None

    def fetch(
        self,
        relative_path: str,
        function: Any,
        *arguments: Any,
        refresh_existing: bool | None = None,
    ) -> bool:
        destination = self.raw_root / relative_path
        should_refresh = (
            self.refresh_existing if refresh_existing is None else refresh_existing
        )
        if destination.exists() and not should_refresh:
            self.skipped_files.append(relative_path)
            print(f"Skipping cached {relative_path}")
            return True
        if self.requests_used >= self.max_requests:
            raise RequestBudgetReached
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.request_delay:
                time.sleep(self.request_delay - elapsed)
        self.requests_used += 1
        print(
            f"[{self.requests_used}/{self.max_requests}] Fetching {relative_path}",
            flush=True,
        )
        try:
            payload = function(*arguments)
            self._last_request_at = time.monotonic()
            _write_json(destination, payload)
            self.saved_files.append(relative_path)
            return True
        except Exception as error:
            self._last_request_at = time.monotonic()
            failure = {
                "path": relative_path,
                "error": f"{type(error).__name__}: {error}",
            }
            self.failures.append(failure)
            print(f"  Failed: {failure['error']}")
            if "TooManyRequests" in type(error).__name__ or "429" in str(error):
                raise GarminRateLimited from error
            return False


def _dates(start: date, end: date) -> list[date]:
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _collect_sleep(client: Any, collector: Collector, days: list[date]) -> None:
    for current in days:
        day = current.isoformat()
        collector.fetch(f"sleep/{day}.json", client.get_sleep_data, day)


def _collect_hrv(client: Any, collector: Collector, days: list[date]) -> None:
    for current in days:
        day = current.isoformat()
        collector.fetch(f"hrv/{day}.json", client.get_hrv_data, day)


def _collect_daily_health(
    client: Any, collector: Collector, start: date, end: date, days: list[date]
) -> None:
    daily_calls = (
        ("daily_summary", client.get_user_summary),
        ("steps", client.get_steps_data),
        ("heart_rate", client.get_heart_rates),
        ("respiration", client.get_respiration_data),
        ("spo2", client.get_spo2_data),
        ("stress", client.get_all_day_stress),
        ("body_battery_events", client.get_body_battery_events),
    )
    for current in days:
        day = current.isoformat()
        for folder, function in daily_calls:
            collector.fetch(f"{folder}/{day}.json", function, day)
    range_name = f"{start.isoformat()}_{end.isoformat()}.json"
    collector.fetch(
        f"body_battery/{range_name}",
        client.get_body_battery,
        start.isoformat(),
        end.isoformat(),
    )


def _collect_body(client: Any, collector: Collector, start: date, end: date) -> None:
    range_name = f"{start.isoformat()}_{end.isoformat()}.json"
    collector.fetch(
        f"body_measurements/{range_name}",
        client.get_weigh_ins,
        start.isoformat(),
        end.isoformat(),
    )


def _collect_activities(
    client: Any, collector: Collector, start: date, end: date
) -> None:
    range_name = f"{start.isoformat()}_{end.isoformat()}.json"
    relative_path = f"activities/{range_name}"
    collector.fetch(
        relative_path,
        client.get_activities_by_date,
        start.isoformat(),
        end.isoformat(),
        None,
        "asc",
    )
    listing_path = collector.raw_root / relative_path
    if not listing_path.exists():
        return
    activities = json.loads(listing_path.read_text(encoding="utf-8")) or []
    if isinstance(activities, dict):
        activities = activities.get("activities") or []
    for activity in activities:
        activity_id = activity.get("activityId")
        if not activity_id:
            continue
        collector.fetch(
            f"activity_details/{activity_id}.json",
            client.get_activity_details,
            str(activity_id),
            refresh_existing=False,
        )
        collector.fetch(
            f"activity_hr_zones/{activity_id}.json",
            client.get_activity_hr_in_timezones,
            str(activity_id),
            refresh_existing=False,
        )


def _collect_training(client: Any, collector: Collector, days: list[date]) -> None:
    latest = days[-1].isoformat()
    collector.fetch(f"max_metrics/{latest}.json", client.get_max_metrics, latest)
    collector.fetch(f"fitness_age/{latest}.json", client.get_fitnessage_data, latest)
    collector.fetch("heart_rate_zones/latest.json", client.get_heart_rate_zones)


def _collect_menstrual(
    client: Any, collector: Collector, start: date, end: date, days: list[date]
) -> None:
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=89), end)
        range_name = f"{chunk_start.isoformat()}_{chunk_end.isoformat()}.json"
        collector.fetch(
            f"menstrual/calendar/{range_name}",
            client.get_menstrual_calendar_data,
            chunk_start.isoformat(),
            chunk_end.isoformat(),
        )
        chunk_start = chunk_end + timedelta(days=1)
    for current in days:
        day = current.isoformat()
        collector.fetch(
            f"menstrual/daily/{day}.json",
            client.get_menstrual_data_for_date,
            day,
        )
    latest = end.isoformat()
    collector.fetch(
        f"menstrual/summary/{latest}.json",
        client.get_menstrual_cycle_summary,
        latest,
    )
    collector.fetch(
        f"menstrual/reports/{latest}.json",
        client.get_menstrual_reports,
        latest,
        6,
    )
    collector.fetch(
        f"menstrual/last_confirmed/{latest}.json",
        client.get_menstrual_last_confirmed,
        latest,
    )


def _run_collectors(
    client: Any, runner: Collector, collectors: tuple[str, ...], start: date, end: date
) -> None:
    days = _dates(start, end)
    for name in collectors:
        if name == "sleep":
            _collect_sleep(client, runner, days)
        elif name == "hrv":
            _collect_hrv(client, runner, days)
        elif name == "daily_health":
            _collect_daily_health(client, runner, start, end, days)
        elif name == "body":
            _collect_body(client, runner, start, end)
        elif name == "activities":
            _collect_activities(client, runner, start, end)
        elif name == "training":
            _collect_training(client, runner, days)
        elif name == "menstrual":
            _collect_menstrual(client, runner, start, end, days)


def _read_state(private_root: Path) -> dict[str, Any]:
    path = private_root / STATE_FILE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def sync(
    windows: list[tuple[date, date, bool]],
    private_root: Path,
    collectors: tuple[str, ...],
    max_requests: int,
    request_delay: float,
    client_path: Path | None = None,
) -> int:
    """Fetch each (start, end, refresh_existing) window in order, sharing one budget."""
    for start, end, _ in windows:
        if start > end:
            raise ValueError("Start date must not be after end date")

    client = _login(private_root, client_path)
    raw_root = private_root / "raw"
    runner = Collector(raw_root, max_requests, request_delay, windows[0][2])
    stopped_reason: str | None = None

    try:
        for start, end, refresh_existing in windows:
            runner.refresh_existing = refresh_existing
            _run_collectors(client, runner, collectors, start, end)
    except RequestBudgetReached:
        stopped_reason = "request_budget_reached"
        print("Request budget reached; stopping before another Garmin request.")
    except GarminRateLimited:
        stopped_reason = "garmin_rate_limited"
        print("Garmin returned a rate limit. Stopping immediately; do not retry now.")

    manifest = {
        "requested_start": min(start for start, _, _ in windows).isoformat(),
        "requested_end": max(end for _, end, _ in windows).isoformat(),
        "windows": [
            {"start": start.isoformat(), "end": end.isoformat(), "refresh_existing": refresh}
            for start, end, refresh in windows
        ],
        "collectors": list(collectors),
        "max_requests": max_requests,
        "requests_used": runner.requests_used,
        "request_delay_seconds": request_delay,
        "saved_files": runner.saved_files,
        "skipped_files": runner.skipped_files,
        "failures": runner.failures,
        "stopped_reason": stopped_reason,
    }
    _write_json(private_root / "last_sync.json", manifest)
    if stopped_reason is None:
        # Every window finished, so everything up to the latest end is fetched.
        # A run cut short doesn't advance this; the next catch-up resumes there.
        state = _read_state(private_root)
        synced_through = max(end for _, end, _ in windows)
        previous = state.get("synced_through")
        if previous and date.fromisoformat(previous) > synced_through:
            synced_through = date.fromisoformat(previous)
        _write_json(
            private_root / STATE_FILE,
            {
                "synced_through": synced_through.isoformat(),
                "last_complete_sync_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
    print(
        f"Saved {len(runner.saved_files)} response file(s) under {raw_root}; "
        f"used {runner.requests_used}/{max_requests} requests."
    )
    if stopped_reason == "garmin_rate_limited" or runner.failures:
        return 1
    return 0


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download local raw Garmin data safely.")
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_COLLECTORS),
        default="daily",
        help="Collector schedule: daily, weekly, or controlled backfill.",
    )
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument(
        "--days",
        type=int,
        default=3,
        help="Number of recent days when --start is omitted (default: 3).",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        choices=COLLECTORS,
        help="Override the profile and run only these collectors.",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        help="Hard request limit; defaults depend on the selected profile.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Minimum seconds between Garmin requests (default: 1.0).",
    )
    parser.add_argument(
        "--refresh-existing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Replace cached responses. Daily/weekly default on; backfill default off.",
    )
    parser.add_argument(
        "--catch-up",
        action="store_true",
        help=(
            "Also fetch days missed since the last complete sync (computer off, "
            "travel), skipping files already saved. Gaps longer than "
            "--catch-up-max-days are left for a manual backfill."
        ),
    )
    parser.add_argument(
        "--catch-up-max-days",
        type=int,
        default=60,
        help="Furthest back --catch-up reaches (default: 60 days).",
    )
    parser.add_argument(
        "--skip-if-synced-within",
        type=float,
        metavar="HOURS",
        help="Exit without contacting Garmin if the last complete sync was this recent.",
    )
    parser.add_argument(
        "--private-root",
        type=Path,
        default=_default_private_root(),
        help="Private directory for tokens and raw responses.",
    )
    parser.add_argument(
        "--client-path",
        type=Path,
        help="Path to the python-garminconnect repository (normally auto-detected).",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if arguments.days < 1:
        raise ValueError("--days must be at least 1")
    if arguments.request_delay < 0:
        raise ValueError("--request-delay cannot be negative")
    end = arguments.end
    start = arguments.start or end - timedelta(days=arguments.days - 1)
    collectors = tuple(arguments.include or PROFILE_COLLECTORS[arguments.profile])
    max_requests = arguments.max_requests or PROFILE_BUDGETS[arguments.profile]
    if max_requests < 1:
        raise ValueError("--max-requests must be at least 1")
    refresh_existing = (
        arguments.refresh_existing
        if arguments.refresh_existing is not None
        else arguments.profile != "backfill"
    )
    client_path = (
        arguments.client_path.expanduser().resolve()
        if arguments.client_path is not None
        else None
    )
    private_root = arguments.private_root.expanduser().resolve()
    state = _read_state(private_root)

    if arguments.skip_if_synced_within is not None and state.get("last_complete_sync_at"):
        last = datetime.fromisoformat(state["last_complete_sync_at"])
        hours = (datetime.now() - last).total_seconds() / 3600
        if hours < arguments.skip_if_synced_within:
            print(f"Last complete sync was {hours:.1f} h ago; nothing to do.")
            return 0

    windows = [(start, end, refresh_existing)]
    if arguments.catch_up and arguments.start is None and state.get("synced_through"):
        gap_start = date.fromisoformat(state["synced_through"]) + timedelta(days=1)
        earliest = start - timedelta(days=arguments.catch_up_max_days)
        if gap_start < earliest:
            print(
                f"Missed days before {earliest.isoformat()} are beyond --catch-up-max-days; "
                "fetch them with --profile backfill --start YYYY-MM-DD."
            )
            gap_start = earliest
        if gap_start < start:
            gap_days = (start - gap_start).days
            # Recent days first (refreshed as usual), then the missed days, where
            # files already saved by an interrupted catch-up are skipped.
            windows.append((gap_start, start - timedelta(days=1), False))
            print(f"Catching up {gap_days} missed day(s) from {gap_start.isoformat()}.")
            if arguments.max_requests is None:
                max_requests += min(
                    CATCH_UP_REQUESTS_PER_DAY * gap_days, CATCH_UP_EXTRA_REQUESTS
                )

    return sync(
        windows,
        private_root,
        collectors,
        max_requests,
        arguments.request_delay,
        client_path,
    )


if __name__ == "__main__":
    raise SystemExit(main())
