from __future__ import annotations
from dataclasses import asdict, dataclass, field
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RAW_FEATURES = ["frp", "brightness", "bright_t31", "scan", "track"]
BASELINE_FEATURES = ["frp_z_cell", "frp_log_ratio_cell", "count_ratio_cell"]
FLAG_FEATURES = ["is_day", "conf_high"]



@dataclass(frozen=True)
class FireAnomalyParams:
    n_estimators: int = 300 
    contamination: float | str = "auto"
    max_samples: int | str = 2048
    random_state: int = 42
    features: tuple[str, ...] = field(default=tuple(RAW_FEATURES + BASELINE_FEATURES + FLAG_FEATURES))


def build_baseline(history: pd.DataFrame) -> pd.DataFrame:
    h = history.assign(month=pd.to_datetime(history["acq_date"]).dt.month)
    g = h.groupby(["h3_r5", "month"])
    base = g["frp"].agg(frp_p50="median", frp_mean="mean", frp_std="std", n_hist="size")
    base["n_days"] = g["acq_date"].nunique()
    base = base.reset_index()
    base["count_per_day"] = base["n_hist"] / base["n_days"]
    fallback_std = base["frp_std"].median() if base["frp_std"].notna().any() else 1.0 
    base["frp_std"] = base["frp_std"].fillna(fallback_std).replace(0, fallback_std)
    return base 


def _confidence_high(conf: pd.Series) -> pd.Series:
    s = conf.astype(str).str.lower()
    viirs_high = s.isin(["h", "high"])
    modis_high = pd.to_numeric(conf, errors="coerce").ge(80).fillna(False)
    return (viirs_high | modis_high).astype(int)


def make_features(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Join detections to their cell/month baseline and derive features.
    Cells without a baseline get neutral values (z=0, ratio=1) so newness alone is not anomalous.
    """
    x = df.assign(month=pd.to_datetime(df["acq_date"]).dt.month)
    x = x.merge(baseline, on=["h3_r5", "month"], how="left")
    x.index = df.index
    x["frp_z_cell"] = ((x["frp"] - x["frp_mean"]) / x["frp_std"]).fillna(0.0)
    x["frp_log_ratio_cell"] = (np.log1p(x["frp"]) - np.log1p(x["frp_p50"].fillna(x["frp"]))).fillna(0.0)
    todays = x.groupby([x["h3_r5"], pd.to_datetime(x["acq_date"]).dt.date])["h3_r5"].transform("size")
    x["count_ratio_cell"] = (todays / x["count_per_day"]).fillna(1.0)
    x["is_day"] = (x["daynight"] == "D").astype(int)
    x["conf_high"] = _confidence_high(x["confidence"])
    return x
def _matrix(features: pd.DataFrame, params: FireAnomalyParams) -> np.ndarray:
    return features[list(params.features)].astype(float).fillna(0.0).to_numpy()
def fit(features: pd.DataFrame, params: FireAnomalyParams = FireAnomalyParams()) -> Pipeline:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "iforest",
                IsolationForest(
                    n_estimators=params.n_estimators,
                    contamination=params.contamination,
                    max_samples=params.max_samples,
                    random_state=params.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(_matrix(features, params))
    return model
def score(model: Pipeline, features: pd.DataFrame, params: FireAnomalyParams = FireAnomalyParams()) -> pd.DataFrame:
    """score (higher = more anomalous), percentile and rank within the input, plus feature deviations."""
    X = _matrix(features, params)
    out = pd.DataFrame(index=features.index)
    out["score"] = -model.score_samples(X)
    out["percentile"] = out["score"].rank(pct=True) * 100
    out["rank_in_day"] = out["score"].rank(ascending=False, method="first").astype(int)
    out["feature_contrib"] = feature_deviations(model, features, params)
    return out
def feature_deviations(model: Pipeline, features: pd.DataFrame, params: FireAnomalyParams) -> pd.Series:
    """Per-feature value and standardized deviation, so the UI can say *why* a point is unusual.
    This is not an attribution method (not SHAP); the model card states that.
    """
    cols = list(params.features)
    values = features[cols].astype(float)
    Z = pd.DataFrame(model.named_steps["scale"].transform(_matrix(features, params)), index=features.index, columns=cols).round(2)
    records = [
        {c: {"value": (None if pd.isna(v) else float(v)), "z": float(z)} for c, v, z in zip(cols, vrow, zrow)}
        for vrow, zrow in zip(values.itertuples(index=False), Z.itertuples(index=False))
    ]
    return pd.Series(records, index=features.index, name="feature_contrib")
def params_dict(params: FireAnomalyParams) -> dict:
    return asdict(params) | {"features": list(params.features), "baseline": "h3_r5 x month"}