from __future__ import annotations
from dataclasses import asdict, dataclass
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler




NUMERIC = ["n_events", "events_per_decade", "median_repose_days", "median_duration_days", "max_vei", "elevation_m"]
CATEGORICAL = ["volcano_type", "tectonic_setting"]


@dataclass(frozen=True)
class VolcanoClusterParams:
    n_clusters: int = 5 
    linkage: str = "ward"


def _prepare(profiles: pd.DataFrame) -> pd.DataFrame:
    p = profiles.copy()
    p[NUMERIC] = p[NUMERIC].astype(float)
    p[NUMERIC] = p[NUMERIC].fillna(p[NUMERIC].median())
    p[CATEGORICAL] = p[CATEGORICAL].fillna("unknown").astype(str)
    return p 


def fit_assign(profiles: pd.DataFrame, params: VolcanoClusterParams = VolcanoClusterParams()) -> tuple[pd.Series, float]:
    """One row per volcano with NUMERIC + CATEGORICAL. Returns labels and silhouette."""
    p = _prepare(profiles)
    pre = ColumnTransformer(
        [("num", StandardScaler(), NUMERIC), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)]
    )
    X = pre.fit_transform(p)
    labels = AgglomerativeClustering(n_clusters=params.n_clusters, linkage=params.linkage).fit_predict(X)
    sil = float(silhouette_score(X, labels)) if 1 < params.n_clusters < len(p) else float("nan")
    return pd.Series(labels, index=profiles.index, name="volcano_cluster"), sil


def describe_clusters(profiles: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    p = _prepare(profiles).assign(cluster=labels)
    g = p.groupby("cluster")
    desc = g[NUMERIC].median()
    desc["n"] = g.size()
    desc["top_type"] = g["volcano_type"].agg(lambda s: s.mode().iat[0])
    desc["top_setting"] = g["tectonic_setting"].agg(lambda s: s.mode().iat[0])
    return desc


def params_dict(params: VolcanoClusterParams, silhouette: float | None = None) -> dict:
    out = asdict(params) | {"numeric": NUMERIC, "categorical": CATEGORICAL}
    if silhouette is not None:
        out["silhouette"] = silhouette
    return out

