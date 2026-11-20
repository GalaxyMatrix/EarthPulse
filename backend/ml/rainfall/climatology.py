"""Per-cell, day-of-year climatology from daily IMERG precipitation."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClimatologyParams:
    window_days: int = 31   # ±15 days around each day-of-year, wrapping at year end
    min_years: int = 3


def build_climatology(daily: pd.DataFrame, params: ClimatologyParams = ClimatologyParams()) -> pd.DataFrame:
    """One row per (cell_id, doy_center): n_obs, mean/std, p50/p90/p99 of daily precip.

    Expects cell_id, obs_date, precip_mm. Cells with fewer than min_years of data are dropped.
    """
    d = daily[["cell_id", "obs_date", "precip_mm"]].copy()
    dt = pd.to_datetime(d["obs_date"])
    d["doy"] = dt.dt.dayofyear
    years = dt.dt.year.groupby(d["cell_id"]).nunique()
    d = d[d["cell_id"].isin(years[years >= params.min_years].index)]
    if d.empty:
        return pd.DataFrame(columns=["cell_id", "doy_center", "window_days", "n_obs", "mean_mm", "std_mm", "p50_mm", "p90_mm", "p99_mm"])

    half = params.window_days // 2
    doy = d["doy"].to_numpy()
    year_len = np.where(pd.to_datetime(d["obs_date"]).dt.is_leap_year, 366, 365)
    parts = []
    for center in range(1, 367):
        dist = np.abs(doy - center)
        mask = np.minimum(dist, year_len - dist) <= half
        if not mask.any():
            continue
        g = d.loc[mask].groupby("cell_id")["precip_mm"]
        stats = g.agg(n_obs="size", mean_mm="mean", std_mm="std")
        q = g.quantile([0.5, 0.9, 0.99]).unstack()
        # Sparse / partial-year inputs can yield an empty unstack for some DOY centers
        if q.empty or q.shape[1] == 0:
            continue
        q = q.rename(columns={0.5: "p50_mm", 0.9: "p90_mm", 0.99: "p99_mm"})
        part = stats.join(q).reset_index()
        part["doy_center"] = center
        parts.append(part)

    if not parts:
        return pd.DataFrame(
            columns=["cell_id", "doy_center", "window_days", "n_obs", "mean_mm", "std_mm", "p50_mm", "p90_mm", "p99_mm"]
        )
    clim = pd.concat(parts, ignore_index=True)
    clim["std_mm"] = clim["std_mm"].fillna(0.0)
    clim["window_days"] = params.window_days
    return clim[["cell_id", "doy_center", "window_days", "n_obs", "mean_mm", "std_mm", "p50_mm", "p90_mm", "p99_mm"]]


def params_dict(params: ClimatologyParams) -> dict:
    return asdict(params)