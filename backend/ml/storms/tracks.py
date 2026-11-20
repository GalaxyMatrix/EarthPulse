from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from ml.common.geo import bearing_deg, circular_mean_deg, haversine_km



@dataclass(frozen=True)
class StormParams:
    ri_threshold_kts: float = 30.0 
    ri_window_hours: float = 24.0 




def track_metrics(points: pd.DataFrame, params: StormParams = StormParams()) -> pd.DataFrame:
    p = points.sort_values("observed_at").reset_index(drop=True).copy()
    t = pd.to_datetime(p["observed_at"], utc=True)
    p["dt_h"] = t.diff().dt.total_seconds() / 3600
    p["dist_km"] = haversine_km(p["lat"].shift(), p["lon"].shift(), p["lat"], p["lon"])
    p["speed_kmh"] = p["dist_km"] / p["dt_h"].replace(0, np.nan)
    p["heading_deg"] = bearing_deg(p["lat"].shift(), p["lon"].shift(), p["lat"], p["lon"])
    p["d_intensity_kts"] = p["magnitude_value"].diff()
    p["intensity_change_window"] = _trailing_change(t, p["magnitude_value"], params.ri_window_hours)
    p["rapid_intensification"] = p["intensity_change_window"] >= params.ri_threshold_kts
    return p




def _trailing_change(t: pd.Series, v: pd.Series, hours: float) -> pd.Series:
    out = pd.Series(np.nan, index=v.index)
    for i in range(len(v)):
        if pd.isna(v.iloc[i]):
            continue
        earlier = (t <= t.iloc[i] - pd.Timedelta(hours=hours)) & v.notna()
        if earlier.any():
            out.iloc[i] = v.iloc[i] - v[earlier].iloc[-1]
    return out


def _duration_h(m: pd.DataFrame) -> float:
    t = pd.to_datetime(m["observed_at"], utc=True)
    return float((t.iloc[-1] - t.iloc[0]).total_seconds() / 3600) if len(m) > 1 else 0.0 


def storm_summary(points: pd.DataFrame, params: StormParams=StormParams()) -> dict:
    m = track_metrics(points, params)
    has_mag = m["magnitude_value"].notna().any()
    return {
        "n_points": int(len(m)),
        "duration_h": _duration_h(m),
        "track_length_km": float(m["dist_km"].sum(skipna=True)),
        "peak_intensity_kts": float(m["magnitude_value"].max()) if has_mag else None,
        "peak_intensity_at": m.loc[m["magnitude_value"].idxmax(), "observed_at"] if has_mag else None,
        "mean_speed_kmh": float(m["speed_kmh"].mean()) if m["speed_kmh"].notna().any() else None,
        "n_ri_points": int(m["rapid_intensification"].sum()),
        "max_window_intensification_kts": float(m["intensity_change_window"].max()) if m["intensity_change_window"].notna().any() else None,
        "magnitude_coverage": float(m["magnitude_value"].notna().mean()),

    }

def track_shape_features(points: pd.DataFrame) -> dict:
    m = track_metrics(points)
    net = float(haversine_km(m["lat"].iloc[0], m["lon"].iloc[0], m["lat"].iloc[-1], m["lon"].iloc[-1]))
    length = float(m["dist_km"].sum(skipna=True))
    return {
        "start_lat": float(m["lat"].iloc[0]),
        "start_lon": float(m["lon"].iloc[0]),
        "end_lat": float(m["lat"].iloc[-1]),
        "end_lon": float(m["lon"].iloc[-1]),
        "net_displacement_km": net,
        "track_length_km": length,
        "sinuosity": length / max(net, 1.0),
        "mean_heading_deg": circular_mean_deg(m["heading_deg"]),
        "peak_intensity_kts": float(m["magnitude_value"].max()) if m["magnitude_value"].notna().any() else np.nan,
        "duration_h": _duration_h(m),

    }


def params_dict(params: StormParams) -> dict:
    return asdict(params)




