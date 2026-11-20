"""NASA EONET v3 events -> events + event_geometries."""
from __future__ import annotations

import json
import logging
from datetime import date

import httpx
import pandas as pd
from shapely.geometry import shape
from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session

from app.db import session_scope
from app.models import Event, EventGeometry, IngestRun
from pipelines.common import date_chunks, ingest_run, point_ewkt, save_raw, upsert
from pipelines.features import to_records
from pipelines.schemas.eonet import EonetEvents, EonetGeometries

log = logging.getLogger(__name__)

SOURCE_KEY = "eonet"
API = "https://eonet.gsfc.nasa.gov/api/v3/events"
CHUNK_DAYS = 365

EVENT_FIELDS = [
    "source_event_id", "category", "title", "description", "link", "is_closed", "closed_at",
    "first_date", "last_date", "lon", "lat", "extra",
]
GEOMETRY_FIELDS = ["source_event_id", "observed_at", "geom_type", "wkt", "magnitude_value", "magnitude_unit"]
EVENT_COLUMNS = [
    "source_id", "ingest_run_id", "source_event_id", "category", "title", "description", "link",
    "is_closed", "closed_at", "first_date", "last_date", "geom", "extra",
]
GEOMETRY_COLUMNS = ["event_id", "observed_at", "geom", "magnitude_value", "magnitude_unit"]


def fetch(client: httpx.Client, start: date, end: date, *, bbox=None) -> bytes:
    params = {"status": "all", "start": start.isoformat(), "end": end.isoformat()}
    if bbox is not None:
        w, s, e, n = bbox
        params["bbox"] = f"{w},{n},{e},{s}"  # EONET wants the upper-left corner, then the lower-right
    resp = client.get(API, params=params)
    resp.raise_for_status()
    return resp.content


def _geom_key(g: dict) -> tuple[str, str]:
    return g["date"], json.dumps(g["coordinates"])


def _wrap_lon(lon: float) -> float:
    """EONET sometimes emits lon in [0, 360) or slightly past ±180; PostGIS/EPSG:4326 wants [-180, 180]."""
    return ((float(lon) + 180.0) % 360.0) - 180.0


def _normalize_coords(coords):
    """Recursively wrap longitude values in a GeoJSON coordinate array."""
    if not coords:
        return coords
    if isinstance(coords[0], (int, float)):
        return [_wrap_lon(coords[0]), *coords[1:]]
    return [_normalize_coords(c) for c in coords]


def merge_events(payloads: list[dict]) -> list[dict]:
    """One entry per event id; an event returned by several request windows keeps the union of its geometries."""
    merged: dict[str, dict] = {}
    for payload in payloads:
        for ev in payload.get("events", []):
            cur = merged.setdefault(ev["id"], {**ev, "geometry": []})
            seen = {_geom_key(g) for g in cur["geometry"]}
            cur["geometry"].extend(g for g in ev.get("geometry") or [] if _geom_key(g) not in seen)
    return list(merged.values())


def _valid_latlon(geom) -> bool:
    """Drop EONET rows with lat outside [-90, 90] (seen on a few bad polygons / swapped coords)."""
    minx, miny, maxx, maxy = geom.bounds
    return -90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0 and -180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0


def parse(events: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(events, geometries) frames, validated. Events without any geometry are skipped."""
    ev_rows, geom_rows = [], []
    for ev in events:
        geoms = sorted(ev.get("geometry") or [], key=lambda g: g["date"])
        if not geoms:
            continue
        pairs = []
        for g in geoms:
            if g.get("type") not in ("Point", "Polygon"):
                continue
            s = shape({"type": g["type"], "coordinates": _normalize_coords(g["coordinates"])})
            if not _valid_latlon(s):
                log.warning("skipping invalid geometry on %s: bounds=%s", ev.get("id"), s.bounds)
                continue
            pairs.append((g, s))
        if not pairs:
            continue
        geoms, shapes = zip(*pairs)
        times = pd.to_datetime([g["date"] for g in geoms], utc=True)
        rep = shapes[0] if shapes[0].geom_type == "Point" else shapes[0].centroid
        ev_rows.append({
            "source_event_id": ev["id"],
            "category": (ev.get("categories") or [{}])[0].get("id"),
            "title": ev.get("title"),
            "description": ev.get("description"),
            "link": ev.get("link"),
            "is_closed": ev.get("closed") is not None,
            "closed_at": ev.get("closed"),
            "first_date": times.min().tz_convert(None).normalize(),
            "last_date": times.max().tz_convert(None).normalize(),
            "lon": _wrap_lon(rep.x),
            "lat": float(rep.y),
            "extra": {"categories": ev.get("categories"), "sources": ev.get("sources")},
        })
        geom_rows += [
            {
                "source_event_id": ev["id"], "observed_at": t, "geom_type": s.geom_type, "wkt": s.wkt,
                "magnitude_value": g.get("magnitudeValue"), "magnitude_unit": g.get("magnitudeUnit"),
            }
            for g, s, t in zip(geoms, shapes, times)
        ]
    events_df = pd.DataFrame(ev_rows, columns=EVENT_FIELDS)
    events_df["closed_at"] = pd.to_datetime(events_df["closed_at"], utc=True)
    geoms_df = pd.DataFrame(geom_rows, columns=GEOMETRY_FIELDS)
    return EonetEvents.validate(events_df), EonetGeometries.validate(geoms_df)


def write(session: Session, events: pd.DataFrame, geoms: pd.DataFrame, run: IngestRun) -> int:
    """Upsert events, then replace their geometries (event_geometries has no natural key)."""
    if events.empty:
        return 0
    ev = events.assign(
        source_id=run.source_id,
        ingest_run_id=run.id,
        first_date=events["first_date"].dt.date,
        last_date=events["last_date"].dt.date,
        geom=[point_ewkt(x, y) for x, y in zip(events["lon"], events["lat"])],
    )
    written = upsert(session, Event, to_records(ev, EVENT_COLUMNS), conflict=["source_id", "source_event_id"])

    ids = dict(
        session.execute(
            select(Event.source_event_id, Event.id).where(
                Event.source_id == run.source_id,
                Event.source_event_id.in_(ev["source_event_id"].tolist()),
            )
        ).all()
    )
    session.execute(delete(EventGeometry).where(EventGeometry.event_id.in_(list(ids.values()))))
    if not geoms.empty:
        g = geoms.assign(event_id=geoms["source_event_id"].map(ids), geom="SRID=4326;" + geoms["wkt"])
        session.execute(insert(EventGeometry.__table__), to_records(g, GEOMETRY_COLUMNS))
    return written


def ingest(start: date, end: date, *, bbox=None) -> IngestRun:
    with ingest_run(SOURCE_KEY, {"start": start, "end": end, "bbox": bbox}) as run:
        payloads = []
        with httpx.Client(timeout=180, follow_redirects=True) as client:
            for lo, hi in date_chunks(start, end, CHUNK_DAYS):
                log.info("EONET %s..%s", lo, hi)
                raw = fetch(client, lo, hi, bbox=bbox)
                run.raw_path = str(save_raw(SOURCE_KEY, f"events_{lo}_{hi}.json", raw).parent)
                payloads.append(json.loads(raw))
        events, geoms = parse(merge_events(payloads))
        run.rows_in = len(events)
        with session_scope() as session:
            run.rows_written = write(session, events, geoms, run)
    return run
