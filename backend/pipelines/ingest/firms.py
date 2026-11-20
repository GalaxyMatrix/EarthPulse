"""NASA FIRMS VIIRS S-NPP 375 m hotspots -> fire_detections, via the area CSV API."""
from __future__ import annotations

import io
import logging
from datetime import date, timedelta

import h3
import httpx
import pandas as pd
from sqlalchemy.orm import Session

from app.config import settings
from app.db import session_scope
from app.models import FireDetection, IngestRun
from pipelines.common import date_chunks, get_region, ingest_run, point_ewkt, save_raw, upsert
from pipelines.features import to_records
from pipelines.schemas.firms import NATURAL_KEY, FirmsDetections

log = logging.getLogger(__name__)

SOURCE_KEY = "firms_viirs_snpp"
BASE = "https://firms.modaps.eosdis.nasa.gov/api"
DEFAULT_SOURCES = ["VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"]
MAX_DAYS = 5  # FIRMS caps DAY_RANGE per area request
H3_RES = 5

CSV_DTYPES = {
    "latitude": float, "longitude": float, "bright_ti4": float, "bright_ti5": float, "frp": float,
    "scan": float, "track": float, "acq_date": str, "acq_time": str, "satellite": str,
    "instrument": str, "confidence": str, "version": str, "daynight": str,
}
COLUMNS = [
    "source_id", "ingest_run_id", "acq_date", "acq_datetime", "lat", "lon", "geom", "brightness",
    "bright_t31", "frp", "scan", "track", "confidence", "daynight", "satellite", "instrument", "version", "h3_r5",
]


def _redact(text: str) -> str:
    key = settings.firms_map_key
    return text.replace(key, "***") if key else text


def _get(client: httpx.Client, url: str) -> str:
    try:
        resp = client.get(url)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(_redact(f"FIRMS request failed: {exc}")) from None
    return resp.text


def ensure_csv(text: str) -> str:
    """FIRMS reports bad keys and rate limits as HTTP 200 with a plain-text message instead of CSV."""
    if not text.lstrip().lower().startswith(("latitude", "data_id")):
        raise RuntimeError(f"FIRMS returned an error instead of CSV: {_redact(text.strip()[:200])!r}")
    return text


def availability(client: httpx.Client) -> dict[str, tuple[date, date]]:
    text = ensure_csv(_get(client, f"{BASE}/data_availability/csv/{settings.firms_map_key}/ALL"))
    df = pd.read_csv(io.StringIO(text), dtype=str)
    return {
        r.data_id: (date.fromisoformat(r.min_date), date.fromisoformat(r.max_date))
        for r in df.itertuples(index=False)
    }


def plan(
    start: date, end: date, sources: list[str], avail: dict[str, tuple[date, date]]
) -> list[tuple[str, date, date]]:
    """Which source covers which days: each source in priority order picks up after the previous one ends,
    so archive (SP) and near-real-time (NRT) never overlap with slightly different coordinates."""
    out, cursor = [], start
    for source in sources:
        if source not in avail:
            continue
        lo, hi = max(cursor, avail[source][0]), min(end, avail[source][1])
        if lo <= hi:
            out.append((source, lo, hi))
            cursor = hi + timedelta(days=1)
    return out


def fetch(client: httpx.Client, source: str, bbox, start: date, days: int) -> str:
    area = ",".join(str(v) for v in bbox)
    url = f"{BASE}/area/csv/{settings.firms_map_key}/{source}/{area}/{days}/{start.isoformat()}"
    return ensure_csv(_get(client, url))


def parse(text: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(ensure_csv(text)), dtype=CSV_DTYPES)
    df = pd.DataFrame({
        "acq_datetime": pd.to_datetime(
            raw["acq_date"] + " " + raw["acq_time"].str.zfill(4), format="%Y-%m-%d %H%M", utc=True
        ),
        "lat": raw["latitude"].round(5),   # fire_detections stores Numeric(8,5); the unique key compares that
        "lon": raw["longitude"].round(5),
        "brightness": raw["bright_ti4"],
        "bright_t31": raw["bright_ti5"],
        "frp": raw["frp"],
        "scan": raw["scan"],
        "track": raw["track"],
        "confidence": raw["confidence"],
        "daynight": raw["daynight"],
        "satellite": raw["satellite"],
        "instrument": raw["instrument"],
        "version": raw["version"],
    })
    df = FirmsDetections.validate(df.drop_duplicates(NATURAL_KEY).reset_index(drop=True))
    df["h3_r5"] = [h3.latlng_to_cell(lat, lon, H3_RES) for lat, lon in zip(df["lat"], df["lon"])]
    return df


def write(session: Session, df: pd.DataFrame, run: IngestRun) -> int:
    rows = df.assign(
        source_id=run.source_id,
        ingest_run_id=run.id,
        acq_date=df["acq_datetime"].dt.date,
        geom=[point_ewkt(lon, lat) for lon, lat in zip(df["lon"], df["lat"])],
    )
    return upsert(session, FireDetection, to_records(rows, COLUMNS), conflict=NATURAL_KEY)


def ingest(region_key: str, start: date, end: date) -> IngestRun:
    if not settings.firms_map_key:
        raise RuntimeError("FIRMS_MAP_KEY is not set (backend/.env)")
    region = get_region(region_key)
    bbox, sources = region["bbox"], region.get("firms_sources", DEFAULT_SOURCES)
    params = {"region": region_key, "start": start, "end": end, "bbox": bbox, "sources": sources}
    with ingest_run(SOURCE_KEY, params) as run:
        frames = []
        with httpx.Client(timeout=120) as client:
            for source, lo, hi in plan(start, end, sources, availability(client)):
                log.info("FIRMS %s %s..%s", source, lo, hi)
                for a, b in date_chunks(lo, hi, MAX_DAYS):
                    text = fetch(client, source, bbox, a, (b - a).days + 1)
                    run.raw_path = str(save_raw(SOURCE_KEY, f"{region_key}_{source}_{a}_{b}.csv", text.encode()).parent)
                    frames.append(parse(text))
        frames = [f for f in frames if not f.empty]
        if not frames:
            run.rows_in = run.rows_written = 0
            return run
        detections = pd.concat(frames, ignore_index=True).drop_duplicates(NATURAL_KEY)
        run.rows_in = len(detections)
        with session_scope() as session:
            run.rows_written = write(session, detections, run)
    return run
