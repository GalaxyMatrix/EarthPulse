from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from app.deps import DbSession, MapBBox, MapDates, as_geojson, envelope
from app.geojson import feature, feature_collection, jsonable
from app.models import RainAnomaly, RainCell, RainDaily

router = APIRouter()


@router.get("/daily")
def list_rain_daily(
    session: DbSession,
    bbox: MapBBox,
    dates: MapDates,
    region: str | None = Query(None),
    limit: int = Query(10_000, ge=1, le=50_000),
):
    """Rain cell polygons with daily precip_mm for days in [start, end]."""
    start, end = dates
    geom_json = as_geojson(RainCell.geom).label("geometry")
    stmt = (
        select(
            RainDaily.id,
            RainDaily.cell_id,
            RainDaily.obs_date,
            RainDaily.precip_mm,
            RainCell.row,
            RainCell.col,
            RainCell.region,
            geom_json,
        )
        .join(RainCell, RainCell.id == RainDaily.cell_id)
        .where(
            RainDaily.obs_date.between(start, end),
            func.ST_Intersects(RainCell.geom, envelope(bbox)),
        )
        .order_by(RainDaily.obs_date.desc(), RainDaily.cell_id)
        .limit(limit)
    )
    if region:
        stmt = stmt.where(RainCell.region == region)

    features = []
    for row in session.execute(stmt).mappings().all():
        props = {k: jsonable(row[k]) for k in row.keys() if k != "geometry"}
        features.append(feature(row["geometry"], props))
    return feature_collection(features)


@router.get("/anomalies")
def list_rain_anomalies(
    session: DbSession,
    bbox: MapBBox,
    dates: MapDates,
    run_id: int | None = Query(None, description="rain_anomaly_z model_runs.id"),
    limit: int = Query(10_000, ge=1, le=50_000),
):
    """Rain cell polygons with anomaly score/z for a model run."""
    start, end = dates
    geom_json = as_geojson(RainCell.geom).label("geometry")
    stmt = (
        select(
            RainAnomaly.id,
            RainAnomaly.cell_id,
            RainAnomaly.model_run_id,
            RainAnomaly.obs_date,
            RainAnomaly.precip_mm,
            RainAnomaly.z,
            RainAnomaly.percentile,
            RainAnomaly.score,
            RainAnomaly.window_days,
            RainAnomaly.matched_event_id,
            RainCell.row,
            RainCell.col,
            RainCell.region,
            geom_json,
        )
        .join(RainCell, RainCell.id == RainAnomaly.cell_id)
        .where(
            RainAnomaly.obs_date.between(start, end),
            func.ST_Intersects(RainCell.geom, envelope(bbox)),
        )
        .order_by(RainAnomaly.score.desc())
        .limit(limit)
    )
    if run_id is not None:
        stmt = stmt.where(RainAnomaly.model_run_id == run_id)

    features = []
    for row in session.execute(stmt).mappings().all():
        props = {k: jsonable(row[k]) for k in row.keys() if k != "geometry"}
        features.append(feature(row["geometry"], props))
    return feature_collection(features)
