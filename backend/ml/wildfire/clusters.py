from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint
from sklearn.cluster import DBSCAN
from ml.common.geo import EARTH_RADIUS_KM, approx_area_km2, km_to_deg



@dataclass(frozen=True)
class ClusterParams:
    eps_km: float = 2.5
    min_samples: int = 3
    hull_buffer_km: float = 0.4


def cluster_detections(df: pd.DataFrame, params: ClusterParams = ClusterParams()) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="int64", index=df.index, name="cluster_label")
    coords = np.radians(df[["lat", "lon"]].to_numpy())
    labels = DBSCAN(
        eps=params.eps_km / EARTH_RADIUS_KM,
        min_samples=params.min_samples,
        metric="haversine",
        algorithm="ball_tree",
    ).fit_predict(coords)
    return pd.Series(labels, index=df.index, name="cluster_label")


def summarize_clusters(df: pd.DataFrame, params: ClusterParams = ClusterParams()) -> pd.DataFrame:
    work = df.assign(cluster_label=cluster_detections(df, params))
    work = work[work["cluster_label"] >= 0]
    has_frp = "frp" in work.columns
    rows = []
    for label, g in work.groupby("cluster_label"):
        pts = MultiPoint(list(zip(g["lon"], g["lat"])))
        hull = pts.convex_hull.buffer(km_to_deg(params.hull_buffer_km))
        rows.append(
            {
                "cluster_label": int(label),
                "hull_wkt": hull.wkt,
                "centroid_wkt": pts.centroid.wkt,
                "lat": float(pts.centroid.y),
                "lon": float(pts.centroid.x),
                "n_detections": int(len(g)),
                "frp_sum": float(g["frp"].sum()) if has_frp else None,
                "frp_max": float(g["frp"].max()) if has_frp else None,
                "area_km2": approx_area_km2(hull),
                "member_index": g.index.tolist(),
            }
        )
    return pd.DataFrame(rows)


def eps_sensitivity(df: pd.DataFrame, eps_values=(1.0, 2.5, 5.0, 10.0), min_samples: int = 3) -> pd.DataFrame:
    rows = []
    for eps in eps_values:
        labels = cluster_detections(df, ClusterParams(eps_km=eps, min_samples=min_samples))
        rows.append(
            {
                "eps_km": eps,
                "n_clusters": int(labels[labels >= 0].nunique()),
                "noise_frac": float((labels == -1).mean()) if len(labels) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def params_dict(params: ClusterParams) -> dict:
    return asdict(params) | {"metric": "haversine", "algorithm": "ball_tree"}

