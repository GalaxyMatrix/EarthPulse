import pandas as pd
import pytest

from ml.landslide.features import NUMERIC_FEATURES, antecedent_rainfall, assemble


@pytest.fixture
def events():
    return pd.DataFrame({"event_id": [1, 2], "lat": [27.7, 27.9], "lon": [85.3, 85.5], "event_date": ["2024-07-31", "2024-07-31"], "trigger": ["downpour", "earthquake"]})


@pytest.fixture
def rainfall():
    dates = pd.date_range("2024-07-01", "2024-08-05")  # includes days after the event
    return pd.concat(
        [
            pd.DataFrame({"event_id": 1, "obs_date": dates, "precip_mm": 10.0}),
            pd.DataFrame({"event_id": 2, "obs_date": dates, "precip_mm": 0.0}),
        ]
    )


def test_trailing_totals_ignore_days_after_event(events, rainfall):
    a = antecedent_rainfall(events, rainfall).set_index("event_id")
    assert a.loc[1, "rain_1d_mm"] == 10.0
    assert a.loc[1, "rain_7d_mm"] == 70.0
    assert a.loc[1, "rain_30d_mm"] == 300.0
    assert a.loc[1, "rain_max_daily_30d"] == 10.0
    assert a.loc[2, "rain_30d_mm"] == 0.0


def test_api_decays_with_time(events, rainfall):
    a = antecedent_rainfall(events, rainfall).set_index("event_id")
    # 31 days of 10 mm with decay 0.9: 10 * (1 - 0.9**31) / 0.1
    assert a.loc[1, "api"] == pytest.approx(10 * (1 - 0.9**31) / 0.1)


def test_assemble_produces_all_model_features(events, rainfall):
    terrain = pd.DataFrame({"event_id": [1, 2], "slope_deg": [35.0, 10.0], "elevation_m": [1500, 900], "relief_m": [400, 100], "land_cover": ["forest", None]})
    x = assemble(events, rainfall, terrain)
    assert set(NUMERIC_FEATURES) <= set(x.columns)
    assert x.loc[x["event_id"] == 1, "slope_x_rain7"].iloc[0] == 35.0 * 70.0
    assert x.loc[x["event_id"] == 2, "land_cover"].iloc[0] == "unknown"
