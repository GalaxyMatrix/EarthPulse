"""Per-volcano event timelines: repose intervals, durations, activity state. Descriptive only."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class TimelineParams:
    active_window_days: int = 90
    min_events_for_stats: int = 3


def volcano_timeline(events: pd.DataFrame) -> pd.DataFrame:
    """Expects start_date, end_date (nullable = ongoing), optional vei."""
    e = events.sort_values("start_date").copy()
    e["start_date"] = pd.to_datetime(e["start_date"])
    e["end_date"] = pd.to_datetime(e["end_date"])
    e["duration_days"] = (e["end_date"] - e["start_date"]).dt.days
    repose = (e["start_date"] - e["end_date"].shift()).dt.days
    start_gap = (e["start_date"] - e["start_date"].shift()).dt.days
    e["repose_days"] = repose.fillna(start_gap).clip(lower=0)
    return e


def volcano_profile(events: pd.DataFrame, reference_date, params: TimelineParams = TimelineParams()) -> dict:
    ref = pd.Timestamp(reference_date)
    t = volcano_timeline(events)
    n = len(t)
    ongoing = bool(t["end_date"].isna().any())
    last_end = t["end_date"].max() if t["end_date"].notna().any() else t["start_date"].max()
    profile = {
        "n_events": int(n),
        "first_event": t["start_date"].min(),
        "last_event": t["start_date"].max(),
        "is_ongoing": ongoing,
        "is_active": ongoing or (ref - last_end).days <= params.active_window_days,
        "days_since_last": None if ongoing else int((ref - last_end).days),
        "median_repose_days": None,
        "median_duration_days": None,
        "events_per_decade": None,
        "max_vei": None,
    }
    if n >= params.min_events_for_stats:
        span_years = max((t["start_date"].max() - t["start_date"].min()).days / 365.25, 1.0)
        profile["median_repose_days"] = float(t["repose_days"].median())
        profile["median_duration_days"] = float(t["duration_days"].median()) if t["duration_days"].notna().any() else None
        profile["events_per_decade"] = float(n / span_years * 10)
    if "vei" in t.columns and t["vei"].notna().any():
        profile["max_vei"] = float(t["vei"].max())
    return profile


def repose_percentile(events: pd.DataFrame, reference_date) -> float | None:
    """Where the current repose sits in this volcano's own repose history (0-100).
    A descriptive comparison, explicitly not an eruption probability."""
    t = volcano_timeline(events)
    hist = t["repose_days"].dropna()
    if len(hist) < 3 or t["end_date"].isna().any():
        return None
    current = (pd.Timestamp(reference_date) - t["end_date"].max()).days
    return float((hist < current).mean() * 100)


def params_dict(params: TimelineParams) -> dict:
    return asdict(params)