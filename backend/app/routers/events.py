from __future__ import annotations

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from app.deps import DbSession, MapBBox, MapDates, as_geojson, envelope
from app.geojson import feature, feature_collection, jsonable
from app.models import Event

router = APIRouter()


@router.get("")
def list_events(
    session: DbSession,
    bbox: MapBBox,
    dates: MapDates,
    category: str | None = Query(None, description="EONET category id, e.g. wildfires"),
    limit: int = Query(5000, ge=1, le=20_000),
):
    """Catalogue events as Point GeoJSON (representative geom)."""
    start, end = dates
    geom_json = as_geojson(Event.geom).label("geometry")
    stmt = (
        select(
            Event.id,
            Event.source_event_id,
            Event.category,
            Event.title,
            Event.description,
            Event.link,
            Event.is_closed,
            Event.first_date,
            Event.last_date,
            geom_json,
        )
        .where(
            Event.first_date <= end,
            Event.last_date >= start,
            func.ST_Intersects(Event.geom, envelope(bbox)),
        )
        .order_by(Event.first_date.desc())
        .limit(limit)
    )
    if category:
        stmt = stmt.where(Event.category == category)

    features = []
    for row in session.execute(stmt).all():
        props = {
            "id": row.id,
            "source_event_id": row.source_event_id,
            "category": row.category,
            "title": row.title,
            "description": row.description,
            "link": row.link,
            "is_closed": row.is_closed,
            "first_date": jsonable(row.first_date),
            "last_date": jsonable(row.last_date),
        }
        features.append(feature(row.geometry, props))
    return feature_collection(features)
