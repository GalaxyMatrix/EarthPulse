from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from ml.common.geo import EARTH_RADIUS_KM



@dataclass(frozen=True)
class LandslideClusterParams:
    eps_km: float = 50.0
    eps_days: float = 7.0 
    min_samples: int = 3 


def cluster_events(events: pd.DataFrame, params: LandslideClusterParams = LandslideClusterParams()) -> pd.Series:
    if events.empty:
        return pd.Series(dtype="int64", index=events.index, name="landslide_cluster")
    lat = np.radians(events["lat"].to_numpy())
    lon = np.radians(events["lon"].to_numpy())
    x = EARTH_RADIUS_KM * lon * np.cos(lat.mean())
    y = EARTH_RADIUS_KM * lat
    t = (pd.to_datetime(events["event_date"]) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    X = np.column_stack([x / params.eps_km, y / params.eps_km, t / params.eps_days])
    labels = DBSCAN(eps=1.0, min_samples=params.min_samples).fit_predict(X)
    return pd.Series(labels, index=events.index, name="landslide_cluster")



def summarize(events: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    w = events.assign(cluster=labels)
    w = w[w["cluster"] >= 0]
    g = w.groupby("cluster")
    out = pd.DataFrame(
        {
            "n_events": g.size(),
            "start": g["event_date"].min(),
            "end": g["event_date"].max(),
            "lat": g["lat"].mean(),
            "lon": g["lon"].mean(),
        }
    )
    if "trigger" in w.columns:
        out["top_trigger"] = g["trigger"].agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
    return out.reset_index()


    
def params_dict(params: LandslideClusterParams) -> dict:
    return asdict(params)
