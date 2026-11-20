import pandas as pd
import pytest

from ml.rainfall.anomaly import RainIForestParams, departures, fit_iforest, rolling_totals, score_iforest, score_z
from ml.rainfall.climatology import build_climatology

STORM_DAY = pd.Timestamp("2023-10-10")


@pytest.fixture
def with_storm(rain_daily):
    d = rain_daily.copy()
    d.loc[(d["cell_id"] == 2) & (d["obs_date"] == STORM_DAY), "precip_mm"] = 250.0
    return d


@pytest.fixture
def dep(with_storm):
    clim = build_climatology(with_storm)
    return departures(with_storm, clim)


def test_rolling_totals():
    daily = pd.DataFrame({"cell_id": 1, "obs_date": pd.date_range("2024-01-01", periods=5), "precip_mm": [1, 2, 3, 4, 5]})
    r = rolling_totals(daily, windows=(1, 3))
    assert r["precip_3d"].isna().sum() == 2
    assert r["precip_3d"].iloc[-1] == 12


def test_storm_day_has_highest_z(dep):
    z = score_z(dep)
    top = z[z["window_days"] == 1].nlargest(1, "z").iloc[0]
    assert top["cell_id"] == 2
    assert pd.Timestamp(top["obs_date"]) == STORM_DAY
    assert top["z"] > 10
    assert top["percentile"] == 99.9


def test_storm_carries_into_multi_day_windows(dep):
    z = score_z(dep)
    week = z[(z["window_days"] == 7) & (z["cell_id"] == 2)]
    peak = week.nlargest(1, "z").iloc[0]
    assert STORM_DAY <= pd.Timestamp(peak["obs_date"]) <= STORM_DAY + pd.Timedelta(days=6)


def test_iforest_ranks_storm_near_top(dep):
    model = fit_iforest(dep, params=RainIForestParams(n_estimators=100))
    s = score_iforest(model, dep)
    assert s["score"].notna().all()
    top10 = s.nlargest(10, "score")
    hit = (top10["cell_id"] == 2) & pd.to_datetime(top10["obs_date"]).between(STORM_DAY, STORM_DAY + pd.Timedelta(days=6))
    assert hit.any()
