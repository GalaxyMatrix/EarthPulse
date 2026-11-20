"""Rainfall inputs: daily cell series, stored climatology, and the anomaly / forecast tables built on them."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ModelRun, RainCell, RainClimatology, RainDaily
from ml.rainfall.anomaly import RainAnomalyParams, departures
from ml.rainfall.forecast import ForecastParams, make_features

CLIMATOLOGY_COLUMNS = ["cell_id", "doy_center", "window_days", "n_obs", "mean_mm", "std_mm", "p50_mm", "p90_mm", "p99_mm"]


def load_cells(session: Session, region: str | None = None) -> pd.DataFrame:
    stmt = select(RainCell.id.label("cell_id"), RainCell.row, RainCell.col, RainCell.region)
    if region is not None:
        stmt = stmt.where(RainCell.region == region)
    return pd.read_sql(stmt, session.connection())


def complete_days(daily: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """One row per cell per calendar day, NaN where data is missing.

    Rolling totals and lags in ml.rainfall count rows, so a gap would otherwise stretch a "7-day" window."""
    idx = pd.MultiIndex.from_product(
        [daily["cell_id"].unique(), pd.date_range(start, end)], names=["cell_id", "obs_date"]
    )
    d = daily.assign(obs_date=pd.to_datetime(daily["obs_date"])).set_index(["cell_id", "obs_date"])
    return d.reindex(idx).reset_index()


def load_daily(
    session: Session, *, start: date, end: date, region: str | None = None, complete: bool = False
) -> pd.DataFrame:
    """cell_id, obs_date, precip_mm in [start, end]. complete=True fills missing days with NaN."""
    stmt = select(RainDaily.cell_id, RainDaily.obs_date, RainDaily.precip_mm).where(
        RainDaily.obs_date.between(start, end)
    )
    if region is not None:
        stmt = stmt.join(RainCell, RainCell.id == RainDaily.cell_id).where(RainCell.region == region)
    daily = pd.read_sql(stmt.order_by(RainDaily.cell_id, RainDaily.obs_date), session.connection())
    return complete_days(daily, start, end) if complete else daily


def load_climatology(session: Session, run_id: int) -> pd.DataFrame:
    stmt = select(*[getattr(RainClimatology, c) for c in CLIMATOLOGY_COLUMNS]).where(
        RainClimatology.model_run_id == run_id
    )
    clim = pd.read_sql(stmt, session.connection())
    if clim.duplicated(["cell_id", "doy_center"]).any():
        raise ValueError(f"Climatology run {run_id} has several window_days per cell/day; use one window per run.")
    return clim


def departures_table(
    session: Session,
    *,
    clim_run_id: int,
    start: date,
    end: date,
    region: str | None = None,
    params: RainAnomalyParams = RainAnomalyParams(),
) -> pd.DataFrame:
    """Per cell/day rolling totals with z and percentile vs climatology, for obs_date in [start, end]."""
    lookback = max(params.windows) - 1
    daily = load_daily(session, start=start - timedelta(days=lookback), end=end, region=region, complete=True)
    dep = departures(daily, load_climatology(session, clim_run_id), params)
    return dep[dep["obs_date"] >= pd.Timestamp(start)].reset_index(drop=True)


def forecast_table(
    session: Session,
    *,
    clim_run_id: int,
    start: date,
    end: date,
    train_end: date,
    region: str | None = None,
    params: ForecastParams = ForecastParams(),
) -> pd.DataFrame:
    """Forecast rows issued in [start, end]; split them with ml.rainfall.forecast.time_split(df, train_end).

    The climatology feature must come from data that ends by train_end, or the test rows see their own years."""
    run = session.get(ModelRun, clim_run_id)
    if run is None:
        raise ValueError(f"No model run {clim_run_id}")
    if run.train_end is not None and run.train_end > train_end:
        raise ValueError(
            f"Climatology run {clim_run_id} was built through {run.train_end}, after train_end {train_end}."
        )
    lookback = max(*params.lags, *params.rolls)
    daily = load_daily(
        session,
        start=start - timedelta(days=lookback),
        end=end + timedelta(days=params.horizon_days),
        region=region,
        complete=True,
    )
    feats = make_features(daily, load_climatology(session, clim_run_id), params)
    return feats[feats["obs_date"].between(pd.Timestamp(start), pd.Timestamp(end))].reset_index(drop=True)
