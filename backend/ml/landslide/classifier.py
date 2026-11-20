from __future__ import annotations
from dataclasses import asdict, dataclass, field
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from ml.landslide.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TRIGGER_MAP = {
    "downpour": "rain", "rain": "rain", "continuous_rain": "rain", "tropical_cyclone": "rain",
    "monsoon": "rain", "flooding": "rain",
    "snowfall_snowmelt": "snowmelt", "earthquake": "earthquake",
    "construction": "human", "mining": "human", "volcano": "volcano", "freeze_thaw": "freeze_thaw",
}



@dataclass(frozen=True)
class LandslideClassifierParams:
    target: str = "trigger_class"
    block_deg: float = 2.0
    n_splits: int = 5
    min_class_support: int = 30
    hgb: dict = field(
        default_factory=lambda: {
            "max_iter": 400,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
            "class_weight": "balanced",
            "random_state": 42,
        }
    )



def prepare_labels(df: pd.DataFrame, params: LandslideClassifierParams = LandslideClassifierParams()) -> pd.DataFrame:
    d = df.copy()
    raw = d["trigger"].astype(str).str.lower().str.strip().str.replace(" ", "_")
    d = d[~raw.isin(["unknown", "nan", "none", ""])]
    d["trigger_class"] = raw.loc[d.index].map(TRIGGER_MAP).fillna("other")
    support = d[params.target].value_counts()
    d.loc[d[params.target].isin(support[support < params.min_class_support].index), params.target] = "other"
    return d 


def spatial_blocks(df: pd.DataFrame, block_deg: float) -> np.ndarray:
    return (
        np.floor(df["lat"] / block_deg).astype(int).astype(str)
        + "_"
        + np.floor(df["lon"] / block_deg).astype(int).astype(str)
    ).to_numpy()


def build_model(params: LandslideClassifierParams = LandslideClassifierParams()) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),

        ]
    )
    return Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(**params.hgb))])

def cross_validate(df: pd.DataFrame, params: LandslideClassifierParams = LandslideClassifierParams()) -> dict:
    """Spatial-block CV so nearby events never sit on both sides of a split."""
    X, y = df[FEATURES], df[params.target]
    groups = spatial_blocks(df, params.block_deg)
    n_splits = min(params.n_splits, len(set(groups)))
    oof = pd.Series(index=df.index, dtype=object)
    fold_f1 = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        model = build_model(params).fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        oof.iloc[te] = pred
        fold_f1.append(float(f1_score(y.iloc[te], pred, average="macro")))
    classes = sorted(y.unique())
    majority = y.mode().iat[0]
    return {
        "macro_f1_folds": fold_f1,
        "macro_f1_mean": float(np.mean(fold_f1)),
        "macro_f1_majority_baseline": float(f1_score(y, [majority] * len(y), average="macro")),
        "accuracy": float((oof == y).mean()),
        "accuracy_majority_baseline": float((y == majority).mean()),
        "classes": classes,
        "confusion_matrix": confusion_matrix(y, oof, labels=classes).tolist(),
        "report": classification_report(y, oof, labels=classes, output_dict=True, zero_division=0),
        "n": int(len(df)),
        "n_blocks": int(len(set(groups))),
        "n_splits": int(n_splits),
    }



def fit(df: pd.DataFrame, params: LandslideClassifierParams = LandslideClassifierParams()) -> Pipeline:
    return build_model(params).fit(df[FEATURES], df[params.target])



def predict(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    proba = model.predict_proba(df[FEATURES])
    out = pd.DataFrame(proba, columns=[f"p_{c}" for c in model.classes_], index=df.index)
    out["pred"] = model.classes_[proba.argmax(axis=1)]
    out["confidence"] = proba.max(axis=1)
    return out



    
def params_dict(params: LandslideClassifierParams) -> dict:
    return asdict(params) | {"features": FEATURES, "trigger_map": TRIGGER_MAP}


