"""Fit per-cell day-of-year rainfall climatology and store it under a ModelRun."""
from __future__ import annotations

import logging
from datetime import date

from app.db import session_scope
from app.models import RainClimatology
from ml.rainfall.climatology import ClimatologyParams, build_climatology, params_dict
from pipelines.common import upsert
from pipelines.features import to_records, write_snapshot
from pipelines.features.rain_climatology import CLIMATOLOGY_COLUMNS, load_daily
from pipelines.runs import create_run

log = logging.getLogger(__name__)

NAME = "rain_climatology"
HAZARD = "rainfall"


def train(
    region_key: str,
    start: date,
    end: date,
    *,
    params: ClimatologyParams | None = None,
    min_years: int | None = None,
) -> int:
    """Build climatology. Default min_years=3; pass min_years=1 for short demos before a full backfill."""
    params = params or ClimatologyParams(min_years=min_years if min_years is not None else 3)
    if min_years is not None and params.min_years != min_years:
        params = ClimatologyParams(window_days=params.window_days, min_years=min_years)

    with session_scope() as session:
        daily = load_daily(session, start=start, end=end, region=region_key)
        if daily.empty:
            raise RuntimeError(f"No rain_daily rows for {region_key} in {start}..{end}")
        clim = build_climatology(daily, params)
        if clim.empty:
            raise RuntimeError(
                f"Climatology empty (need >= {params.min_years} years per cell); "
                f"have years in range or pass --min-years 1"
            )
        _, digest = write_snapshot(clim, NAME)
        run = create_run(
            session,
            name=NAME,
            hazard=HAZARD,
            region=region_key,
            params=params_dict(params),
            train_start=start,
            train_end=end,
            data_manifest_hash=digest,
            notes="Per-cell DOY ±window climatology from IMERG daily precip",
            registry_alias="challenger",
        )
        rows = clim.assign(model_run_id=run.id)
        # unique key is (cell_id, doy_center, window_days) — not per run; upsert refreshes stats
        n = upsert(
            session,
            RainClimatology,
            to_records(rows, ["cell_id", "model_run_id", "doy_center", "window_days",
                               "n_obs", "mean_mm", "std_mm", "p50_mm", "p90_mm", "p99_mm"]),
            conflict=["cell_id", "doy_center", "window_days"],
        )
        log.info("rain climatology run %s: %d rows (upsert rc=%s)", run.id, len(clim), n)
        return run.id
