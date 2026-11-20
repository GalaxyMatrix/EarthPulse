import numpy as np
import pandas as pd
import pytest

from ml.common.evaluation import forecast_report
from ml.rainfall.climatology import build_climatology
from ml.rainfall.forecast import (
    BASELINES,
    ForecastParams,
    feature_columns,
    fit_lgbm,
    make_features,
    predict_lgbm,
    rolling_origin_splits,
    time_split,
)

TRAIN_END = "2022-12-31"


@pytest.fixture
def feats(rain_daily):
    return make_features(rain_daily, build_climatology(rain_daily))


def test_lag_and_target_alignment(rain_daily, feats):
    cell = feats[feats["cell_id"] == 1].reset_index(drop=True)
    raw = rain_daily[rain_daily["cell_id"] == 1].reset_index(drop=True)
    i = 100
    assert cell.loc[i, "lag_1"] == raw.loc[i, "precip_mm"]
    assert cell.loc[i, "lag_2"] == raw.loc[i - 1, "precip_mm"]
    assert cell.loc[i, "target"] == raw.loc[i + 1, "precip_mm"]


def test_time_split_has_no_leakage(feats):
    train, test = time_split(feats, TRAIN_END)
    end = pd.Timestamp(TRAIN_END)
    assert train["target_date"].max() <= end
    assert test["obs_date"].min() > end
    assert train["target"].notna().all() and test["target"].notna().all()


def test_rolling_origin_folds_expand(feats):
    folds = list(rolling_origin_splits(feats, "2021-12-31", n_folds=3, fold_days=90))
    sizes = [len(train) for _, train, _ in folds]
    assert sizes == sorted(sizes) and len(set(sizes)) == 3


def test_baselines_and_lgbm_run_and_report(feats):
    params = ForecastParams(lgbm={**ForecastParams().lgbm, "n_estimators": 50})
    train, test = time_split(feats, TRAIN_END)
    preds = {name: fn(test).to_numpy() for name, fn in BASELINES.items()}
    model = fit_lgbm(train, params)
    preds["rain_forecast_lgbm"] = predict_lgbm(model, test, params).to_numpy()

    assert all(len(p) == len(test) for p in preds.values())
    assert (preds["rain_forecast_lgbm"] >= 0).all()

    rows = forecast_report(test["target"].to_numpy(), preds, baseline="rain_forecast_climatology")
    metrics = {(r["model"], r["metric"]) for r in rows}
    assert ("rain_forecast_lgbm", "skill_vs_baseline") in metrics
    assert ("rain_forecast_climatology", "skill_vs_baseline") not in metrics
    assert all(np.isfinite(r["value"]) for r in rows if r["metric"] in ("mae", "rmse"))


def test_feature_columns_exist(feats):
    assert set(feature_columns()) <= set(feats.columns)
