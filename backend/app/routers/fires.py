from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from app.deps import DbSession, MapBBox, MapDates, as_geojson, envelope
from app.geojson import feature, feature_collection, jsonable
from app.models import FireAnomalyScore, FireCluster, FireDetection

router = APIRouter()


@router.get("/fires")
def list_fires(
    session: DbSession,
    bbox: MapBBox,
    dates: MapDates,
    run_id: int | None = Query(None, description="Optional fire_anomaly_if model_runs.id to join scores"),
    limit: int = Query(5000, ge=1, le=20_000),
):
    """FIRMS detections as Point GeoJSON; optional anomaly score from run_id."""
    start, end = dates
    geom_json = as_geojson(FireDetection.geom).label("geometry")
    cols = [
        FireDetection.id,
        FireDetection.acq_date,
        FireDetection.acq_datetime,
        FireDetection.lat,
        FireDetection.lon,
        FireDetection.frp,
        FireDetection.brightness,
        FireDetection.bright_t31,
        FireDetection.confidence,
        FireDetection.daynight,
        FireDetection.satellite,
        FireDetection.instrument,
        FireDetection.h3_r5,
        geom_json,
    ]
    if run_id is not None:
        stmt = (
            select(
                *cols,
                FireAnomalyScore.score,
                FireAnomalyScore.percentile,
                FireAnomalyScore.rank_in_day,
            )
            .select_from(FireDetection)
            .join(
                FireAnomalyScore,
                (FireAnomalyScore.detection_id == FireDetection.id)
                & (FireAnomalyScore.model_run_id == run_id),
            )
            .where(
                FireDetection.acq_date.between(start, end),
                func.ST_Intersects(FireDetection.geom, envelope(bbox)),
            )
        )
    else:
        stmt = select(*cols).where(
            FireDetection.acq_date.between(start, end),
            func.ST_Intersects(FireDetection.geom, envelope(bbox)),
        )
    stmt = stmt.order_by(FireDetection.acq_datetime.desc()).limit(limit)

    features = []
    for row in session.execute(stmt).mappings().all():
        props = {k: jsonable(row[k]) for k in row.keys() if k != "geometry"}
        features.append(feature(row["geometry"], props))
    return feature_collection(features)


@router.get("/fire-clusters")
def list_fire_clusters(
    session: DbSession,
    bbox: MapBBox,
    dates: MapDates,
    run_id: int | None = Query(None, description="fire_clusters_dbscan model_runs.id"),
    limit: int = Query(2000, ge=1, le=10_000),
):
    """Daily DBSCAN cluster hulls as Polygon GeoJSON."""
    start, end = dates
    geom_json = as_geojson(FireCluster.hull).label("geometry")
    stmt = (
        select(
            FireCluster.id,
            FireCluster.model_run_id,
            FireCluster.cluster_date,
            FireCluster.cluster_label,
            FireCluster.n_detections,
            FireCluster.frp_sum,
            FireCluster.frp_max,
            FireCluster.area_km2,
            FireCluster.matched_event_id,
            geom_json,
        )
        .where(
            FireCluster.cluster_date.between(start, end),
            func.ST_Intersects(FireCluster.hull, envelope(bbox)),
        )
        .order_by(FireCluster.cluster_date.desc(), FireCluster.frp_sum.desc().nullslast())
        .limit(limit)
    )
    if run_id is not None:
        stmt = stmt.where(FireCluster.model_run_id == run_id)

    features = []
    for row in session.execute(stmt).mappings().all():
        props = {k: jsonable(row[k]) for k in row.keys() if k != "geometry"}
        features.append(feature(row["geometry"], props))
    return feature_collection(features)
