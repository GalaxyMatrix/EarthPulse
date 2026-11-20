from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


WINDOWS = (1, 3, 7)

@dataclass(frozen=True)
class RainAnomalyParams:
    windows: tuple[int, ...] = WINDOWS
    std_floor_mm: float = 1.0 



@dataclass(frozen=True)
class RainIForestParams:
    n_estimators: int = 300
    contamination: float | str = "auto"
    max_samples: int | str = 4096
    random_state: int = 42 


def rolling_totals(daily: pd.DataFrame, windows: tuple[int, ...] = WINDOWS) -> pd.DataFrame:
    d = daily.sort_values(["cell_id", "obs_date"]).copy()
    g = d.groupby("cell_id")["precip_mm"]
    for w in windows:
        d[f"precip_{w}d"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=w).sum())
    return d 


def _percentile_vs_band(v, p50, p90, p99) -> pd.Series:
    """Piecewise-linear percentile from three climatology quantiles, capped at 99.9."""
    pct = pd.Series(np.nan, index=v.index, dtype=float)
    lo = v <= p50
    pct[lo] = 50 * v[lo] / p50[lo].replace(0, np.nan)
    mid = (v > p50) & (v <= p90)
    pct[mid] = 50 + 40 * (v[mid] - p50[mid]) / (p90[mid] - p50[mid]).replace(0, np.nan)
    hi = (v > p90) & (v <= p99)
    pct[hi] = 90 + 9 * (v[hi] - p90[hi]) / (p99[hi] - p90[hi]).replace(0, np.nan)
    pct[v > p99] = 99.9
    # dry day in a cell whose median is 0 mm: neither anomalous nor informative
    return pct.fillna(50.0).clip(0, 99.9)



def departures(daily: pd.DataFrame, climatology: pd.DataFrame, params: RainAnomalyParams = RainAnomalyParams()) -> pd.DataFrame:
    d = rolling_totals(daily, params.windows)
    d["doy_center"] = pd.to_datetime(d["obs_date"]).dt.dayofyear
    m = d.merge(climatology, on=["cell_id", "doy_center"], how="inner")
    for w in params.windows:
        v = m[f"precip_{w}d"]
        mean_w = m["mean_mm"]  * w
        std_w = np.maximum(m["std_mm"] * np.sqrt(w), params.std_floor_mm)
        m[f"z_{w}d"] = (v - mean_w) / std_w
        m[f"pct_{w}d"] = _percentile_vs_band(v, m["p50_mm"] * w, m["p90_mm"] * w, m["p99_mm"] * w)
    return m 


def score_z(dep: pd.DataFrame, params: RainAnomalyParams = RainAnomalyParams()) -> pd.DataFrame:
    parts = [] 
    for w in params.windows:
        x = dep.dropna(subset=[f"precip_{w}d"])
        parts.append(
            pd.DataFrame({
                  "cell_id": x["cell_id"],
                    "obs_date": x["obs_date"],
                    "window_days": w,
                    "precip_mm": x[f"precip_{w}d"],
                    "z": x[f"z_{w}d"],
                    "percentile": x[f"pct_{w}d"],
                    "score": x[f"z_{w}d"],

            })
        )
    
    return pd.concat(parts, ignore_index=True)



def iforest_features(params: RainAnomalyParams = RainAnomalyParams()) -> list[str]:
    return [f"z_{w}d" for w in params.windows] + [f"pct_{w}d" for w in params.windows]


def fit_iforest(dep: pd.DataFrame, anom: RainAnomalyParams = RainAnomalyParams(), params: RainIForestParams = RainIForestParams()) -> IsolationForest:
    X = dep[iforest_features(anom)].dropna().to_numpy()
    return IsolationForest(
        n_estimators=params.n_estimators,
        contamination=params.contamination,
        max_samples=params.max_samples,
        random_state=params.random_state,
        n_jobs=-1,
    ).fit(X)



def score_iforest(model: IsolationForest, dep: pd.DataFrame, anom: RainAnomalyParams = RainAnomalyParams()) -> pd.DataFrame:
    """One row per (cell_id, obs_date) with complete windows. Only wet departures count:
    unusually *dry* periods are flagged by the forest too, but are out of scope here."""
    cols = iforest_features(anom)
    x = dep.dropna(subset=cols)
    raw = -model.score_samples(x[cols].to_numpy())
    wet = (x[[f"z_{w}d" for w in anom.windows]] > 0).any(axis=1).to_numpy()
    longest = max(anom.windows)
    return pd.DataFrame(
        {
            "cell_id": x["cell_id"].to_numpy(),
            "obs_date": x["obs_date"].to_numpy(),
            "window_days": longest,
            "precip_mm": x[f"precip_{longest}d"].to_numpy(),
            "z": x[f"z_{longest}d"].to_numpy(),
            "percentile": x[f"pct_{longest}d"].to_numpy(),
            "score": np.where(wet, raw, np.nan),
        }
    ).dropna(subset=["score"])



def params_dict(anom: RainAnomalyParams, iforest: RainIForestParams | None = None) -> dict:
    out = {"anomaly": asdict(anom), "method": "z_vs_doy_climatology"}
    if iforest is not None:
        out |= {"iforest": asdict(iforest), "features": iforest_features(anom), "method": "iforest_on_departures"}
    return out
