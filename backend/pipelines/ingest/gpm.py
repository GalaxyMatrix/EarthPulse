"""GPM IMERG Final daily precipitation (GPM_3IMERGDF V07) -> rain_cells + rain_daily, via OPeNDAP subsets."""
from __future__ import annotations

import logging
import math
from datetime import date

import httpx
import netCDF4
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import session_scope
from app.models import IngestRun, RainCell, RainDaily
from pipelines.common import get_region, ingest_run, point_ewkt, save_raw, upsert
from pipelines.features import to_records
from pipelines.schemas.imerg import ImergDaily

log = logging.getLogger(__name__)

SOURCE_KEY = "gpm_imerg_daily"
RES = 0.1
OPENDAP = "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGDF.07"
FILENAME = "3B-DAY.MS.MRG.3IMERG.{day:%Y%m%d}-S000000-E235959.V07B.nc4"
WRITE_EVERY = 30  # days per transaction, so an interrupted backfill keeps what it already loaded
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def cell_indices(bbox) -> tuple[range, range]:
    """IMERG (row, col) index ranges covering bbox; rows count 0.1° bands up from -90°, cols east from -180°."""
    w, s, e, n = bbox
    rows = range(math.floor(round((s + 90) / RES, 6)), math.ceil(round((n + 90) / RES, 6)))
    cols = range(math.floor(round((w + 180) / RES, 6)), math.ceil(round((e + 180) / RES, 6)))
    return rows, cols


def cell_rows(region_key: str, bbox) -> list[dict]:
    rows, cols = cell_indices(bbox)
    out = []
    for r in rows:
        for c in cols:
            w, s = round(c * RES - 180, 2), round(r * RES - 90, 2)
            e, n = round(w + RES, 2), round(s + RES, 2)
            out.append({
                "row": r,
                "col": c,
                "region": region_key,
                "geom": f"SRID=4326;POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))",
                "centroid": point_ewkt(round(w + RES / 2, 3), round(s + RES / 2, 3)),
            })
    return out


def ensure_cells(region_key: str) -> int:
    bbox = get_region(region_key)["rain"]["bbox"]
    with session_scope() as session:
        return upsert(session, RainCell, cell_rows(region_key, bbox), conflict=["row", "col"])


def cell_lookup(session: Session, region_key: str) -> dict[tuple[int, int], int]:
    stmt = select(RainCell.id, RainCell.row, RainCell.col).where(RainCell.region == region_key)
    return {(r, c): i for i, r, c in session.execute(stmt)}


def loaded_days(session: Session, region_key: str, start: date, end: date) -> set[date]:
    stmt = (
        select(RainDaily.obs_date)
        .join(RainCell, RainCell.id == RainDaily.cell_id)
        .where(RainCell.region == region_key, RainDaily.obs_date.between(start, end))
        .distinct()
    )
    return set(session.scalars(stmt))


def fetch(client: httpx.Client, day: date, rows: range, cols: range, *, attempts: int = 5) -> bytes | None:
    """The day's regional subset as netCDF-4, or None if GES DISC has not published that day yet."""
    import time

    r0, r1, c0, c1 = rows.start, rows.stop - 1, cols.start, cols.stop - 1
    url = (
        f"{OPENDAP}/{day:%Y}/{day:%m}/{FILENAME.format(day=day)}.nc4"
        f"?precipitation[0:0][{c0}:{c1}][{r0}:{r1}],lat[{r0}:{r1}],lon[{c0}:{c1}]"
    )
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = client.get(url)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            if not resp.content.startswith(HDF5_SIGNATURE):
                raise RuntimeError(
                    f"GES DISC returned {resp.headers.get('content-type')} instead of netCDF for {day}; "
                    "check EARTHDATA_TOKEN and that 'NASA GESDISC DATA ARCHIVE' is authorized in your Earthdata profile"
                )
            return resp.content
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
            wait = min(60, 2 ** attempt)
            log.warning("IMERG %s attempt %d/%d failed (%s); retry in %ds", day, attempt, attempts, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"IMERG fetch failed for {day} after {attempts} attempts") from last_exc


def parse(raw: bytes, *, obs_date: date, rows: range, cols: range) -> pd.DataFrame:
    with netCDF4.Dataset("subset.nc4", mode="r", memory=raw) as ds:
        precip = np.ma.filled(ds["precipitation"][0, :, :].astype("float64"), np.nan)  # IMERG axes: (lon, lat)
        lat = np.asarray(ds["lat"][:], dtype="float64")
        lon = np.asarray(ds["lon"][:], dtype="float64")
    want_lat = np.array(rows) * RES - 90 + RES / 2
    want_lon = np.array(cols) * RES - 180 + RES / 2
    if (
        precip.shape != (len(cols), len(rows))
        or not np.allclose(lat, want_lat, atol=1e-3)
        or not np.allclose(lon, want_lon, atol=1e-3)
    ):
        raise ValueError(f"IMERG subset for {obs_date} does not line up with the requested grid")
    col_idx, row_idx = np.meshgrid(np.array(cols), np.array(rows), indexing="ij")
    df = pd.DataFrame({
        "row": row_idx.ravel(),
        "col": col_idx.ravel(),
        "obs_date": pd.Timestamp(obs_date),
        "precip_mm": precip.ravel(),
    })
    df = df[df["precip_mm"] >= 0]  # drops NaN and negative fill values; rain_daily.precip_mm is NOT NULL
    return ImergDaily.validate(df.reset_index(drop=True))


def write(session: Session, df: pd.DataFrame, cell_ids: dict[tuple[int, int], int], run: IngestRun) -> int:
    rows = df.assign(
        cell_id=[cell_ids[(r, c)] for r, c in zip(df["row"], df["col"])],
        ingest_run_id=run.id,
        obs_date=df["obs_date"].dt.date,
    )
    return upsert(
        session, RainDaily, to_records(rows, ["cell_id", "ingest_run_id", "obs_date", "precip_mm"]),
        conflict=["cell_id", "obs_date"],
    )


def ingest(region_key: str, start: date, end: date, *, force: bool = False) -> IngestRun:
    if not settings.earthdata_token:
        raise RuntimeError("EARTHDATA_TOKEN is not set (backend/.env)")
    rows, cols = cell_indices(get_region(region_key)["rain"]["bbox"])
    ensure_cells(region_key)
    params = {"region": region_key, "start": start, "end": end, "force": force}
    with ingest_run(SOURCE_KEY, params) as run:
        with session_scope() as session:
            cell_ids = cell_lookup(session, region_key)
            done = set() if force else loaded_days(session, region_key, start, end)
        days = [d for d in pd.date_range(start, end).date if d not in done]
        log.info("IMERG %s: %d days to fetch, %d already loaded", region_key, len(days), len(done))
        run.rows_in = run.rows_written = 0
        batch: list[pd.DataFrame] = []

        def flush() -> None:
            if not batch:
                return
            df = pd.concat(batch, ignore_index=True)
            with session_scope() as session:
                run.rows_written += write(session, df, cell_ids, run)
            run.rows_in += len(df)
            log.info("IMERG %s: loaded through %s", region_key, df["obs_date"].max().date())
            batch.clear()

        headers = {"Authorization": f"Bearer {settings.earthdata_token}"}
        with httpx.Client(headers=headers, timeout=120, follow_redirects=True) as client:
            for day in days:
                raw = fetch(client, day, rows, cols)
                if raw is None:
                    log.warning("IMERG %s not published yet; skipped", day)
                    continue
                run.raw_path = str(save_raw(SOURCE_KEY, f"{region_key}_{day:%Y%m%d}.nc4", raw).parent)
                batch.append(parse(raw, obs_date=day, rows=rows, cols=cols))
                if len(batch) >= WRITE_EVERY:
                    flush()
            flush()
    return run
