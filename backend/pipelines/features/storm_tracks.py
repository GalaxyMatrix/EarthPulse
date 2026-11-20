from __future__ import annotations
from datetime import date
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from app.models import Event, EventGeometry, Source
from ml.storms.clustering import StormClusterParams, storm_feature_table
from pipelines.evaluate import envelope


CATEGORY = "severeStorms"
TO_KTS = {"kts": 1.0, "kt": 1.0, "knots": 1.0, "m/s": 1.943844, "km/h": 0.539957, "mph": 0.868976}


def normalize_intensity(points: pd.DataFrame) -> pd.DataFrame:
    p = points.copy()
    unit = p["magnitude_unit"].astype("string").str.strip().str.lower()
    p["magnitude_value"] = p["magnitude_value"] * unit.map(TO_KTS).astype(float)
    return p.drop(columns="magnitude_unit")




def load_track_points(
    session: Session, *, source_key: str, start: date | None = None, end: date | None = None, bbox=None
) -> pd.DataFrame:
    """One row per track point: event_id, event_geometry_id, observed_at, lat, lon, magnitude_value (kts).
    source_key is required because EONET and IBTrACS both describe the same storms."""
    point = func.ST_PointOnSurface(EventGeometry.geom)
    stmt = (
        select(
            EventGeometry.event_id,
            EventGeometry.id.label("event_geometry_id"),
            EventGeometry.observed_at,
            func.ST_Y(point).label("lat"),
            func.ST_X(point).label("lon"),
            EventGeometry.magnitude_value,
            EventGeometry.magnitude_unit,
        )
        .join(Event, Event.id == EventGeometry.event_id)
        .join(Source, Source.id == Event.source_id)
        .where(Event.category == CATEGORY, Source.key == source_key)
        .order_by(EventGeometry.event_id, EventGeometry.observed_at)
    )
    if start is not None:
        stmt = stmt.where(Event.last_date >= start)
    if end is not None:
        stmt = stmt.where(Event.first_date <= end)
    if bbox is not None:
        stmt = stmt.where(func.ST_Intersects(Event.geom, envelope(bbox)))
    points = pd.read_sql(stmt, session.connection())
    return normalize_intensity(points.drop_duplicates(["event_id", "observed_at"]))




def tracks_by_event(points: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {int(eid): g.reset_index(drop=True) for eid, g in points.groupby("event_id", sort=False)}




    
def load_feature_table(
    session: Session,
    *,
    source_key: str,
    start: date | None = None,
    end: date | None = None,
    bbox=None,
    params: StormClusterParams = StormClusterParams(),
) -> pd.DataFrame:
    """One row per storm with ml.storms.clustering.FEATURES, plus first_observed_at for time-based splits."""
    tracks = tracks_by_event(load_track_points(session, source_key=source_key, start=start, end=end, bbox=bbox))
    table = storm_feature_table(tracks, params)
    if table.empty:
        return table
    table["first_observed_at"] = table["event_id"].map({eid: t["observed_at"].min() for eid, t in tracks.items()})
    return table


