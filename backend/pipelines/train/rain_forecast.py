"""Next-day rainfall forecast experiment: baselines + LightGBM, with skill metrics."""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from app.db import session_scope
from app.models import ModelRun, RainForecast
from ml.common.evaluation import forecast_report
from ml.rainfall.forecast import (
    BASELINES,
    ForecastParams,
    fit_lgbm,
    params_dict,
    predict_lgbm,
    time_split,
)
from pipelines.common import upsert
from pipelines.evaluate import write_evaluations
from pipelines.features import to_records, write_snapshot
from pipelines.features.rain_climatology import forecast_table
from pipelines.runs import create_run, save_artifact

log = logging.getLogger(__name__)

HAZARD = "rainfall"
LGBM_NAME = "rain_forecast_lgbm"


def train(
    clim_run_id: int,
    *,
    start: date,
    end: date,
    train_end: date,
    params: ForecastParams = ForecastParams(),
) -> dict[str, int]:
    """Fit LGBM on issues in [start, train_end]; score test issues after train_end through end.

    Creates one ModelRun per predictor (baselines + lgbm) and writes rain_forecasts + evaluations
    on the LGBM run (skill vs persistence).
    """
    with session_scope() as session:
        clim = session.get(ModelRun, clim_run_id)
        if clim is None:
            raise ValueError(f"No climatology run {clim_run_id}")
        region = clim.region
        feats = forecast_table(
            session,
            clim_run_id=clim_run_id,
            start=start,
            end=end,
            train_end=train_end,
            region=region,
            params=params,
        )
        if feats.empty:
            raise RuntimeError("Forecast feature table is empty")
        train_df, test_df = time_split(feats, train_end)
        if train_df.empty or test_df.empty:
            raise RuntimeError(
                f"Bad split at {train_end}: train={len(train_df)} test={len(test_df)}; "
                "need observations on both sides of train_end"
            )
        _, digest = write_snapshot(train_df.reset_index(drop=True), LGBM_NAME)
        model = fit_lgbm(train_df, params)

        run_ids: dict[str, int] = {}
        preds: dict[str, object] = {}

        for name, fn in BASELINES.items():
            pred = fn(test_df)
            preds[name] = pred.to_numpy()
            run = create_run(
                session,
                name=name,
                hazard=HAZARD,
                region=region,
                params=params_dict(params) | {"climatology_run_id": clim_run_id, "role": "baseline"},
                train_start=start,
                train_end=train_end,
                data_manifest_hash=digest,
                notes=f"Forecast baseline ({name})",
            )
            run_ids[name] = run.id
            _write_forecasts(session, run.id, test_df, pred)

        lgbm_pred = predict_lgbm(model, test_df, params)
        preds[LGBM_NAME] = lgbm_pred.to_numpy()
        lgbm_run = create_run(
            session,
            name=LGBM_NAME,
            hazard=HAZARD,
            region=region,
            params=params_dict(params) | {"climatology_run_id": clim_run_id},
            train_start=start,
            train_end=train_end,
            data_manifest_hash=digest,
            notes="LightGBM tweedie next-day precip",
            registry_alias="challenger",
        )
        lgbm_run.artifact_path = save_artifact(
            LGBM_NAME, lgbm_run.version, {"model": model, "params": params_dict(params)}
        )
        run_ids[LGBM_NAME] = lgbm_run.id
        _write_forecasts(session, lgbm_run.id, test_df, lgbm_pred)

        y = test_df["target"].to_numpy()
        rows = []
        for r in forecast_report(y, preds, baseline="rain_forecast_persistence"):
            rows.append({
                "metric": f"{r['model']}.{r['metric']}" if r["model"] != LGBM_NAME else r["metric"],
                "value": r["value"],
                "baseline": r.get("baseline"),
                "details": {**(r.get("details") or {}), "model": r["model"], "n_test": int(len(test_df))},
            })
        write_evaluations(
            session, lgbm_run, rows,
            eval_start=pd.Timestamp(test_df["obs_date"].min()).date(),
            eval_end=pd.Timestamp(test_df["obs_date"].max()).date(),
        )
        log.info("forecast runs %s; test n=%d", run_ids, len(test_df))
        return run_ids


def _write_forecasts(session, run_id: int, test_df: pd.DataFrame, pred: pd.Series) -> None:
    rows = pd.DataFrame({
        "cell_id": test_df["cell_id"].to_numpy(),
        "model_run_id": run_id,
        "issue_date": pd.to_datetime(test_df["obs_date"]).dt.date,
        "target_date": pd.to_datetime(test_df["target_date"]).dt.date,
        "pred_mm": pred.to_numpy(),
        "actual_mm": test_df["target"].to_numpy(),
    })
    upsert(
        session,
        RainForecast,
        to_records(rows, ["cell_id", "model_run_id", "issue_date", "target_date", "pred_mm", "actual_mm"]),
        conflict=["cell_id", "issue_date", "target_date", "model_run_id"],
    )
