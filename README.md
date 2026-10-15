# garminxsleep

A local dashboard that merges historical Sleepmeter exports with Garmin
Connect sleep, HRV, activity, and health data, then charts
how training load, bedtime regularity, sleep debt, sleep aids, and the menstrual
cycle relate to sleep.

It ships with a **synthetic dataset**, so it runs without any accounts or
personal data.

## Run it

Python 3.12 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
streamlit run dashboard/app.py
```

The dashboard opens on the sample data in `sample_data/`: about three months
of invented Sleepmeter history and three and a half months of invented Garmin data, with two
weeks of overlap.

## How the data is merged

- **Before the Garmin start date**, nights come from Sleepmeter. Garmin
  nights in that window that start and end within two minutes of a Sleepmeter
  night are matched to it, and the Garmin sleep stages are mapped onto the
  Sleepmeter session. Any part of the night Garmin did not cover is marked
  "stage unavailable" rather than guessed.
- **From the Garmin start date**, Garmin is authoritative.
- Garmin timestamps are UTC and are shown in the configured time zone.
  Sleepmeter interruptions ("holes") reduce asleep time without changing the
  session's start and end.
- Optional JSON override files correct individual nights by hand.

The merged dataset is also exported to `data/sleep_dashboard.duckdb` for SQL
analysis.

## Using your own data

Point `GARMINXSLEEP_DATA_DIR` at a folder with the same layout as
`sample_data/`. The Git-ignored `private/` folder is the intended place:

```text
private/
├── settings.json              {"timezone": "Area/City", "garmin_start_date": "YYYY-MM-DD"}
├── sleep_history_*.csv        Sleepmeter export (the newest file is used)
├── aid_colors.json            optional chart colour per sleep aid
├── garmin_*_overrides.json    optional manual corrections
└── raw/<collector>/*.json     written by the Garmin sync below
```

```bash
export GARMINXSLEEP_DATA_DIR="$PWD/private"
streamlit run dashboard/app.py
```

Without `garmin_start_date`, Garmin takes over from its first recorded night.

### Sync Garmin data

Install the optional client and fetch the last three days:

```bash
python -m pip install -e ".[sync]"
python -m dashboard.garmin_sync
```

The first run asks for your Garmin email, password, and MFA code. The password
is not saved; a reusable token and the raw JSON responses are written to
`GARMINXSLEEP_DATA_DIR` (or `private/`). Requests are sequential, at least one
second apart, and capped per run. A Garmin `429` stops the run immediately.

```bash
python -m dashboard.garmin_sync --profile weekly                 # body, VO2 max, cycle data
python -m dashboard.garmin_sync --start 2026-01-01 --end 2026-01-31 --profile backfill
python -m dashboard.garmin_sync --include sleep --days 7          # one collector
```

Collectors: `sleep`, `hrv`, `daily_health`, `body`, `activities`, `training`,
`menstrual`.

## Tests

```bash
pytest
```

The tests run on `sample_data/` and check the merge rules: session and daily
totals agree, no night is counted twice, Garmin naps become sessions, nights
Garmin missed fall back to Sleepmeter, and the Overview charts share a date
window and hour scale.

## Regenerating the sample data

```bash
python scripts/generate_sample_data.py
```

The generator is seeded, so the output is identical on every run.
