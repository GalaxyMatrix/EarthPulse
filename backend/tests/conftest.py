import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def fire_history(rng):
    """Two years of ordinary VIIRS-like detections in three H3 cells."""
    n = 3000
    return pd.DataFrame(
        {
            "h3_r5": rng.choice(["85a", "85b", "85c"], n),
            "acq_date": pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D"),
            "lat": 38.0 + rng.random(n),
            "lon": 23.0 + rng.random(n),
            "frp": rng.gamma(2.0, 5.0, n),
            "brightness": rng.normal(330, 8, n),
            "bright_t31": rng.normal(295, 4, n),
            "scan": rng.uniform(0.35, 0.6, n),
            "track": rng.uniform(0.35, 0.6, n),
            "daynight": rng.choice(["D", "N"], n),
            "confidence": rng.choice(["l", "n", "h"], n),
        }
    )


@pytest.fixture
def rain_daily(rng):
    """Five years of gamma-distributed daily precipitation for three cells, gap-free."""
    dates = pd.date_range("2019-01-01", "2023-12-31")
    return pd.concat(
        [
            pd.DataFrame({"cell_id": c, "obs_date": dates, "precip_mm": rng.gamma(0.6, 4.0, len(dates))})
            for c in (1, 2, 3)
        ],
        ignore_index=True,
    )


def make_track(rng, start_lat, start_lon, dlat, dlon, n=12, base_kts=40.0, growth_kts=5.0):
    """A storm track with 6-hourly points moving by (dlat, dlon) per step."""
    t = pd.date_range("2024-08-01", periods=n, freq="6h", tz="UTC")
    return pd.DataFrame(
        {
            "observed_at": t,
            "lat": start_lat + dlat * np.arange(n) + rng.normal(0, 0.02, n),
            "lon": start_lon + dlon * np.arange(n) + rng.normal(0, 0.02, n),
            "magnitude_value": base_kts + growth_kts * np.arange(n),
        }
    )
