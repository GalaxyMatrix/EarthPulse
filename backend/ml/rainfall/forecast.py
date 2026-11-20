"""Next-day rainfall forecasting experiment: baselines vs LightGBM with a strict time split."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastParams:
    horizon_days: int = 1
    lags: tuple[int, ...] = (1, 2, 3, 5, 7, 14)
    rolls: tuple[int, ...] = (3, 7, 14, 30)
    lgbm: dict = field(
        default_factory=lambda: {
            "n_estimators": 600,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 50,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "objective": "tweedie",   # zero-inflated, skewed target
            "random_state": 42,
            "verbose": -1,
        }
    )


def make_features(daily: pd.DataFrame, climatology: pd.DataFrame, params: ForecastParams = ForecastParams()) -> pd.DataFrame:
    """Issue date = obs_date; target = precip at obs_date + horizon. lag_1 is the issue day itself."""
    d = daily.sort_values(["cell_id", "obs_date"]).copy()
    d["obs_date"] = pd.to_datetime(d["obs_date"])
    g = d.groupby("cell_id")["precip_mm"]
    for lag in params.lags:
        d[f"lag_{lag}"] = g.shift(lag - 1)
    for w in params.rolls:
        d[f"roll_{w}"] = g.transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
    d["target"] = g.shift(-params.horizon_days)
    d["target_date"] = d["obs_date"] + pd.Timedelta(days=params.horizon_days)
    d["doy_center"] = d["target_date"].dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * d["doy_center"] / 366)
    d["doy_cos"] = np.cos(2 * np.pi * d["doy_center"] / 366)
    clim = climatology[["cell_id", "doy_center", "mean_mm", "p90_mm"]].rename(columns={"mean_mm": "clim_mean", "p90_mm": "clim_p90"})
    return d.merge(clim, on=["cell_id", "doy_center"], how="left")


def feature_columns(params: ForecastParams = ForecastParams()) -> list[str]:
    return [f"lag_{l}" for l in params.lags] + [f"roll_{w}" for w in params.rolls] + ["doy_sin", "doy_cos", "clim_mean", "clim_p90"]


def time_split(df: pd.DataFrame, train_end) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train: target observed on or before train_end. Test: issued after train_end.
    No row has information from both sides of the boundary."""
    train_end = pd.Timestamp(train_end)
    train = df[df["target_date"] <= train_end].dropna(subset=["target"])
    test = df[df["obs_date"] > train_end].dropna(subset=["target"])
    return train, test


def rolling_origin_splits(df: pd.DataFrame, first_train_end, n_folds: int = 4, fold_days: int = 90):
    """Expanding-window folds for a more robust estimate than a single split."""
    start = pd.Timestamp(first_train_end)
    for i in range(n_folds):
        end = start + pd.Timedelta(days=i * fold_days)
        train, test = time_split(df, end)
        yield end, train, test[test["obs_date"] <= end + pd.Timedelta(days=fold_days)]


def predict_persistence(df: pd.DataFrame) -> pd.Series:
    return df["lag_1"].rename("pred_mm")


def predict_climatology(df: pd.DataFrame) -> pd.Series:
    return df["clim_mean"].fillna(df["roll_30"]).rename("pred_mm")


def predict_rolling_mean(df: pd.DataFrame, w: int = 7) -> pd.Series:
    return df[f"roll_{w}"].rename("pred_mm")


def fit_lgbm(train: pd.DataFrame, params: ForecastParams = ForecastParams()):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(**params.lgbm)
    model.fit(train[feature_columns(params)], train["target"].clip(lower=0))
    return model


def predict_lgbm(model, df: pd.DataFrame, params: ForecastParams = ForecastParams()) -> pd.Series:
    pred = model.predict(df[feature_columns(params)])
    return pd.Series(np.clip(pred, 0, None), index=df.index, name="pred_mm")


BASELINES = {
    "rain_forecast_persistence": predict_persistence,
    "rain_forecast_climatology": predict_climatology,
    "rain_forecast_roll7": predict_rolling_mean,
}


def params_dict(params: ForecastParams) -> dict:
    return asdict(params) | {"features": feature_columns(params), "baselines": list(BASELINES)}