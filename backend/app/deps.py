"""FastAPI dependencies: DB session and shared map query params."""
from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from typing import Annotated

from fastapi import Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db import SessionLocal

BBox = tuple[float, float, float, float]  # west, south, east, north


def get_db() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


DbSession = Annotated[Session, Depends(get_db)]


def parse_bbox(bbox: str = Query(..., description="west,south,east,north (EPSG:4326)")) -> BBox:
    try:
        parts = [float(x.strip()) for x in bbox.split(",")]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="bbox must be four comma-separated numbers") from exc
    if len(parts) != 4:
        raise HTTPException(status_code=422, detail="bbox must be west,south,east,north")
    w, s, e, n = parts
    if not (-180 <= w <= 180 and -180 <= e <= 180 and -90 <= s <= 90 and -90 <= n <= 90):
        raise HTTPException(status_code=422, detail="bbox coordinates out of range")
    if w >= e or s >= n:
        raise HTTPException(status_code=422, detail="bbox requires west<east and south<north")
    return w, s, e, n


def date_range(
    start: date = Query(..., description="First day (inclusive)"),
    end: date = Query(..., description="Last day (inclusive)"),
) -> tuple[date, date]:
    if start > end:
        raise HTTPException(status_code=422, detail="start must be on or before end")
    return start, end


MapBBox = Annotated[BBox, Depends(parse_bbox)]
MapDates = Annotated[tuple[date, date], Depends(date_range)]


def envelope(bbox: BBox):
    w, s, e, n = bbox
    return func.ST_MakeEnvelope(w, s, e, n, 4326)


def as_geojson(geom_col):
    return func.ST_AsGeoJSON(geom_col)
