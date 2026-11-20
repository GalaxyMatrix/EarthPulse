from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from ml.storms.tracks import track_shape_features
FEATURES = [
    "start_lat", "start_lon", "net_displacement_km", "track_length_km", "sinuosity",
    "heading_sin", "heading_cos", "peak_intensity_kts", "duration_h",
]



@dataclass(frozen=True)
class StormClusterParams:
    k_range: tuple[int, ...] = (3,4,5,6,8)
    min_points: int = 3
    random_state: int = 42 



@dataclass
class StormClusterModel:
    scaler: StandardScaler
    kmeans: KMeans
    k : int
    silhouette: float
    fill_values: dict 


def storm_feature_table(tracks: dict, params: StormClusterParams = StormClusterParams()) -> pd.DataFrame:
    rows = [] 
    for event_id, pts in tracks.items():
        if len(pts) < params.min_points:
            continue
        f = track_shape_features(pts)
        h = np.radians(f.pop("mean_heading_deg"))
        f["heading_sin"] = float(np.sin(h)) if not np.isnan(h) else 0.0 
        f["heading_cos"] = float(np.cos(h)) if not np.isnan(h) else 0.0 
        rows.append({"event_id": event_id, **f})
    return pd.DataFrame(rows)


def fit(table: pd.DataFrame, params: StormClusterParams = StormClusterParams()) -> StormClusterModel:
    fill = table[FEATURES].median().to_dict()
    X = table[FEATURES].fillna(fill)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    best: StormClusterModel | None = None
    for k in params.k_range:
        if k >= len(table):
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=params.random_state).fit(Xs)
        s = float(silhouette_score(Xs, km.labels_))
        if best is None or s > best.silhouette:
            best = StormClusterModel(scaler, km, k, s, fill)
    if best is None:
        raise ValueError(f"Not enough storms ({len(table)}) for k in {params.k_range}")
    return best


def assign(model: StormClusterModel, table: pd.DataFrame) -> pd.Series:
    X = model.scaler.transform(table[FEATURES].fillna(model.fill_values))
    return pd.Series(model.kmeans.predict(X), index=table.index, name="storm_cluster")


def describe_clusters(table: pd.DataFrame, labels:pd.Series) -> pd.DataFrame:
    desc = table.assign(cluster=labels).groupby("cluster")[FEATURES].median()
    desc["n"] = labels.value_counts().sort_index()
    return desc


def nearest_storms(model:StormClusterModel, table: pd.DataFrame, event_id, n: int = 5) -> pd.DataFrame:
    Xs = model.scaler.transform(table[FEATURES].fillna(model.fill_values))
    i = table.index[table["event_id"] == event_id][0]
    d = np.linalg.norm(Xs - Xs[table.index.get_loc(i)], axis = 1)
    return table.assign(distance=d).drop(index=i).nsmallest(n, "distance")[["event_id", "distance"]]


def params_dict(params: StormClusterParams, model: StormClusterModel | None = None) -> dict:
    out = asdict(params) | {"features": FEATURES}
    if model is not None:
        out |= {"chosen_k": model.k, "silhouette": model.silhouette}
    return out

