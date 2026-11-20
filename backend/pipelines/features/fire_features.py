from __future__ import annotations

from datetime import date, timedelta

import h3
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import FireDetection
from ml.wildfire.anomaly import build_baseline, make_features
from pipelines.evaluate import envelope

H3_RES = 5
BASELINE_DAYS = 730


def fill_h3(df: pd.DataFrame) -> pd.DataFrame:
    """Compute h3_r5 for rows the ingest left empty; the baseline is keyed on it."""
    missing = df["h3_r5"].isna()
    if not missing.any():
        return df
    df = df.copy()
    df.loc[missing, "h3_r5"] = [
        h3.latlng_to_cell(lat, lon, H3_RES) for lat, lon in zip(df.loc[missing, "lat"], df.loc[missing, "lon"])
    ]
    return df


def _filters(stmt, *, bbox, instrument: str):
    # VIIRS and MODIS brightness/FRP sit on different scales, so a baseline never mixes instruments
    stmt = stmt.where(FireDetection.instrument == instrument)
    if bbox is not None:
        stmt = stmt.where(func.ST_Intersects(FireDetection.geom, envelope(bbox)))
    return stmt


def load_detections(
    session: Session, *, start: date, end: date, bbox=None, instrument: str = "VIIRS"
) -> pd.DataFrame:
    """Detections in [start, end] with every column ml.wildfire needs, plus ids for writing scores back."""
    stmt = select(
        FireDetection.id.label("detection_id"),
        FireDetection.acq_date,
        FireDetection.acq_datetime,
        FireDetection.lat,
        FireDetection.lon,
        FireDetection.frp,
        FireDetection.brightness,
        FireDetection.bright_t31,
        FireDetection.scan,
        FireDetection.track,
        FireDetection.confidence,
        FireDetection.daynight,
        FireDetection.satellite,
        FireDetection.instrument,
        FireDetection.h3_r5,
    ).where(FireDetection.acq_date.between(start, end))
    stmt = _filters(stmt, bbox=bbox, instrument=instrument)
    return fill_h3(pd.read_sql(stmt, session.connection()))


def load_history(
    session: Session, *, before: date, days: int = BASELINE_DAYS, bbox=None, instrument: str = "VIIRS"
) -> pd.DataFrame:
    """h3_r5, acq_date, frp for the `days` before `before` (exclusive): input for build_baseline."""
    stmt = select(
        FireDetection.h3_r5,
        FireDetection.lat,
        FireDetection.lon,
        FireDetection.acq_date,
        FireDetection.frp,
    ).where(FireDetection.acq_date >= before - timedelta(days=days), FireDetection.acq_date < before)
    stmt = _filters(stmt, bbox=bbox, instrument=instrument)
    return fill_h3(pd.read_sql(stmt, session.connection()))[["h3_r5", "acq_date", "frp"]]


def load_training_table(
    session: Session,
    *,
    start: date,
    end: date,
    baseline_days: int = BASELINE_DAYS,
    bbox=None,
    instrument: str = "VIIRS",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features for [start, end] against a baseline built strictly before start.

    Returns (features, baseline). Save the baseline with the model: scoring later days must reuse it
    via make_features(load_detections(...), baseline) rather than rebuilding it from newer data."""
    history = load_history(session, before=start, days=baseline_days, bbox=bbox, instrument=instrument)
    baseline = build_baseline(history)
    detections = load_detections(session, start=start, end=end, bbox=bbox, instrument=instrument)
    return make_features(detections, baseline), baseline
