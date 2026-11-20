from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.models import LandslideEvent, LandslideFeature, RainCell, RainDaily
from ml.landslide.features import WINDOWS, LandslideFeatureParams, add_derived, assemble
from pipelines.evaluate import envelope
from pipelines.features import to_records

FEATURE_VERSION = "v1"
RAIN_SOURCE = "rain_daily"
TERRAIN = ["slope_deg", "elevation_m", "relief_m", "land_cover"]
STORED = [f"rain_{w}d_mm" for w in WINDOWS] + ["rain_max_daily_30d", "api"] + TERRAIN
BATCH = 2000  # keeps each INSERT under Postgres' 65,535 bind-parameter limit


def load_events(session: Session, *, start: date | None = None, end: date | None = None, bbox=None) -> pd.DataFrame:
    stmt = select(
        LandslideEvent.id.label("event_id"),
        LandslideEvent.lat,
        LandslideEvent.lon,
        LandslideEvent.event_date,
        LandslideEvent.trigger,
        LandslideEvent.location_accuracy,
    )
    if start is not None:
        stmt = stmt.where(LandslideEvent.event_date >= start)
    if end is not None:
        stmt = stmt.where(LandslideEvent.event_date <= end)
    if bbox is not None:
        stmt = stmt.where(func.ST_Intersects(LandslideEvent.geom, envelope(bbox)))
    return pd.read_sql(stmt, session.connection())


def load_rainfall_from_cells(session: Session, events: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Daily rain from the tracked IMERG cell containing each event, up to and including the event date.

    Events outside every tracked cell get no rows, so their rain features stay NaN instead of 0."""
    empty = pd.DataFrame({"event_id": pd.Series(dtype="int64"), "obs_date": [], "precip_mm": pd.Series(dtype="float64")})
    if events.empty:
        return empty
    stmt = (
        select(
            LandslideEvent.id.label("event_id"),
            RainCell.id.label("cell_id"),
            RainDaily.obs_date,
            RainDaily.precip_mm,
        )
        .join(RainCell, func.ST_Contains(RainCell.geom, LandslideEvent.geom))
        .join(RainDaily, RainDaily.cell_id == RainCell.id)
        .where(
            LandslideEvent.id.in_(events["event_id"].tolist()),
            RainDaily.obs_date <= LandslideEvent.event_date,
            RainDaily.obs_date > LandslideEvent.event_date - lookback_days,
        )
    )
    r = pd.read_sql(stmt, session.connection())
    if r.empty:
        return empty
    # an event on a shared cell edge matches two cells; keep one so totals aren't doubled
    r = r[r["cell_id"] == r["event_id"].map(r.groupby("event_id")["cell_id"].min())]
    return r.drop(columns="cell_id").astype({"event_id": "int64", "precip_mm": "float64"})


def no_terrain(events: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {"event_id": events["event_id"], "slope_deg": np.nan, "elevation_m": np.nan, "relief_m": np.nan, "land_cover": None}
    )


def build(
    events: pd.DataFrame,
    rainfall: pd.DataFrame,
    terrain: pd.DataFrame | None = None,
    params: LandslideFeatureParams = LandslideFeatureParams(),
) -> pd.DataFrame:
    x = assemble(events, rainfall, terrain if terrain is not None else no_terrain(events), params)
    x["rain_days"] = x["event_id"].map(rainfall.groupby("event_id").size()).fillna(0).astype(int)
    return x


def write(session: Session, features: pd.DataFrame, *, feature_version: str = FEATURE_VERSION) -> int:
    """Upsert one LandslideFeature row per event for this feature_version. Commit is left to the caller."""
    rows = []
    for r in to_records(features, ["event_id", "rain_days", *STORED]):
        event_id, rain_days = r.pop("event_id"), r.pop("rain_days")
        rows.append({
            "landslide_event_id": event_id,
            "feature_version": feature_version,
            "extra": {"rain_source": RAIN_SOURCE, "rain_days": rain_days},
            **r,
        })
    for i in range(0, len(rows), BATCH):
        stmt = insert(LandslideFeature).values(rows[i : i + BATCH])
        stmt = stmt.on_conflict_do_update(
            constraint="uq_landslide_features_event_version",
            set_={c: stmt.excluded[c] for c in [*STORED, "extra"]},
        )
        session.execute(stmt)
    return len(rows)


def run(
    session: Session,
    *,
    start: date | None = None,
    end: date | None = None,
    bbox=None,
    terrain: pd.DataFrame | None = None,
    params: LandslideFeatureParams = LandslideFeatureParams(),
    feature_version: str = FEATURE_VERSION,
) -> int:
    events = load_events(session, start=start, end=end, bbox=bbox)
    if events.empty:
        return 0
    rainfall = load_rainfall_from_cells(session, events, lookback_days=max(params.windows))
    return write(session, build(events, rainfall, terrain, params), feature_version=feature_version)


def load_table(session: Session, feature_version: str = FEATURE_VERSION) -> pd.DataFrame:
    """Stored features + event columns, with derived features recomputed: input for ml.landslide.classifier."""
    stmt = (
        select(
            LandslideEvent.id.label("event_id"),
            LandslideEvent.lat,
            LandslideEvent.lon,
            LandslideEvent.event_date,
            LandslideEvent.trigger,
            *[getattr(LandslideFeature, c) for c in STORED],
        )
        .join(LandslideFeature, LandslideFeature.landslide_event_id == LandslideEvent.id)
        .where(LandslideFeature.feature_version == feature_version)
    )
    return add_derived(pd.read_sql(stmt, session.connection()))