from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
WINDOWS = (1, 3, 7, 14, 30)
NUMERIC_FEATURES = [f"rain_{w}d_mm" for w in WINDOWS] + [
    "rain_max_daily_30d", "api", "slope_deg", "elevation_m", "relief_m", "doy_sin", "doy_cos", "slope_x_rain7",
]
CATEGORICAL_FEATURES = ["land_cover"]


@dataclass(frozen=True)
class LandslideFeatureParams:
    windows: tuple[int, ...] = WINDOWS
    api_decay: float = 0.9   # antecedent precipitation index decay per day
def antecedent_rainfall(events: pd.DataFrame, rainfall: pd.DataFrame, params: LandslideFeatureParams = LandslideFeatureParams()) -> pd.DataFrame:
    r = rainfall.merge(events[["event_id", "event_date"]], on="event_id")
    r["obs_date"] = pd.to_datetime(r["obs_date"])
    r["event_date"] = pd.to_datetime(r["event_date"])
    r = r[r["obs_date"] <= r["event_date"]]
    r["days_before"] = (r["event_date"] - r["obs_date"]).dt.days
    out = pd.DataFrame({"event_id": r["event_id"].unique()}).set_index("event_id")
    for w in params.windows:
        out[f"rain_{w}d_mm"] = r[r["days_before"] < w].groupby("event_id")["precip_mm"].sum()
    out["rain_max_daily_30d"] = r[r["days_before"] < 30].groupby("event_id")["precip_mm"].max()
    r["api_term"] = r["precip_mm"] * params.api_decay ** r["days_before"]
    out["api"] = r.groupby("event_id")["api_term"].sum()
    return out.fillna(0.0).reset_index()
def params_dict(params: LandslideFeatureParams) -> dict:
    return asdict(params) | {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES}



def add_derived(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    x["event_date"] = pd.to_datetime(x["event_date"])
    doy = x["event_date"].dt.dayofyear
    x["doy_sin"] = np.sin(2 * np.pi * doy / 366)
    x["doy_cos"] = np.cos(2 * np.pi * doy / 366)
    x["slope_x_rain7"] = x["slope_deg"] * x["rain_7d_mm"]
    x["land_cover"] = x["land_cover"].fillna("unknown").astype(str)
    return x


def assemble(events: pd.DataFrame, rainfall: pd.DataFrame, terrain: pd.DataFrame, params: LandslideFeatureParams = LandslideFeatureParams()) -> pd.DataFrame:
    x = events.merge(antecedent_rainfall(events, rainfall, params), on="event_id", how="left")
    x = x.merge(terrain, on="event_id", how="left")
    return add_derived(x)