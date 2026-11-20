"""Score rainfall anomalies vs a climatology ModelRun (z-score method)."""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from app.db import session_scope
from app.models import ModelRun, RainAnomaly
from ml.rainfall.anomaly import RainAnomalyParams, params_dict, score_z
from pipelines.common import upsert
from pipelines.features import to_records
from pipelines.features.rain_climatology import departures_table
from pipelines.runs import create_run

log = logging.getLogger(__name__)

NAME = "rain_anomaly_z"
HAZARD = "rainfall"
# DB unique key is (cell_id, obs_date, model_run_id) — one window per run
DEFAULT_WINDOW = 7


def score(
    clim_run_id: int,
    start: date,
    end: date,
    *,
    window_days: int = DEFAULT_WINDOW,
    params: RainAnomalyParams = RainAnomalyParams(),
) -> int:
    with session_scope() as session:
        clim = session.get(ModelRun, clim_run_id)
        if clim is None or clim.name != "rain_climatology":
            raise ValueError(f"Run {clim_run_id} is not a rain_climatology model run")
        region = clim.region
        dep = departures_table(
            session, clim_run_id=clim_run_id, start=start, end=end, region=region, params=params
        )
        if dep.empty:
            raise RuntimeError(f"No departures for clim run {clim_run_id} in {start}..{end}")
        scored = score_z(dep, params)
        scored = scored[scored["window_days"] == window_days].copy()
        if scored.empty:
            raise RuntimeError(f"No anomaly rows for window_days={window_days}")

        run = create_run(
            session,
            name=NAME,
            hazard=HAZARD,
            region=region,
            params=params_dict(params) | {"window_days": window_days, "climatology_run_id": clim_run_id},
            train_start=clim.train_start,
            train_end=clim.train_end,  # eval must start after this
            notes=f"Z-score vs climatology run {clim_run_id}; stored window={window_days}d",
            registry_alias="challenger",
        )
        rows = scored.assign(model_run_id=run.id)
        rows["obs_date"] = pd.to_datetime(rows["obs_date"]).dt.date
        n = upsert(
            session,
            RainAnomaly,
            to_records(
                rows,
                ["cell_id", "model_run_id", "obs_date", "precip_mm", "z", "percentile", "score", "window_days"],
            ),
            conflict=["cell_id", "obs_date", "model_run_id"],
        )
        log.info("rain anomaly run %s: %d rows (upsert rc=%s)", run.id, len(rows), n)
        return run.id
