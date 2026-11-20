import numpy as np
import pandas as pd

from ml.storms.tracks import storm_summary, track_metrics, track_shape_features
from tests.conftest import make_track


def test_rapid_intensification_flagged(rng):
    # +5 kt per 6 h = +20 kt / 24 h: below threshold
    slow = track_metrics(make_track(rng, 15, -50, 0.3, -0.5, growth_kts=5))
    assert not slow["rapid_intensification"].any()
    # +10 kt per 6 h = +40 kt / 24 h: rapid intensification
    fast = track_metrics(make_track(rng, 15, -50, 0.3, -0.5, growth_kts=10))
    assert fast["rapid_intensification"].any()
    assert fast["intensity_change_window"].max() == 40.0


def test_speed_and_heading(rng):
    pts = make_track(rng, 10, -40, 0.0, -0.5, n=5).assign(lat=10.0, lon=lambda d: -40 - 0.5 * np.arange(5))
    m = track_metrics(pts)
    # 0.5 deg longitude at 10N ~ 54.7 km per 6 h ~ 9.1 km/h, heading due west
    assert np.allclose(m["speed_kmh"].iloc[1:], 9.1, atol=0.2)
    assert np.allclose(m["heading_deg"].iloc[1:], 270, atol=0.5)


def test_summary_handles_missing_magnitude(rng):
    pts = make_track(rng, 15, -50, 0.3, -0.5)
    pts.loc[pts.index[::2], "magnitude_value"] = np.nan
    s = storm_summary(pts)
    assert s["n_points"] == 12
    assert s["duration_h"] == 66.0
    assert s["magnitude_coverage"] == 0.5
    assert s["peak_intensity_kts"] is not None


def test_no_magnitude_at_all(rng):
    s = storm_summary(make_track(rng, 15, -50, 0.3, -0.5).assign(magnitude_value=np.nan))
    assert s["peak_intensity_kts"] is None
    assert s["n_ri_points"] == 0


def test_shape_features_straight_track_has_low_sinuosity(rng):
    f = track_shape_features(make_track(rng, 15, -50, 0.5, -0.5))
    assert 1.0 <= f["sinuosity"] < 1.1
    assert f["net_displacement_km"] > 500
    assert 270 <= f["mean_heading_deg"] <= 360
