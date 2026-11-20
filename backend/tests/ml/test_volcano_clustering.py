import numpy as np
import pandas as pd

from ml.volcano.clustering import VolcanoClusterParams, describe_clusters, fit_assign, params_dict


def profiles(rng):
    n = 15
    frequent = pd.DataFrame(
        {
            "n_events": rng.integers(80, 120, n), "events_per_decade": rng.uniform(15, 25, n),
            "median_repose_days": rng.uniform(50, 150, n), "median_duration_days": rng.uniform(5, 20, n),
            "max_vei": rng.integers(1, 3, n).astype(float), "elevation_m": rng.uniform(1000, 1500, n),
            "volcano_type": "shield", "tectonic_setting": "intraplate",
        }
    )
    rare = pd.DataFrame(
        {
            "n_events": rng.integers(2, 6, n), "events_per_decade": rng.uniform(0.1, 0.5, n),
            "median_repose_days": rng.uniform(5000, 9000, n), "median_duration_days": rng.uniform(100, 400, n),
            "max_vei": rng.integers(4, 6, n).astype(float), "elevation_m": rng.uniform(3000, 4000, n),
            "volcano_type": "stratovolcano", "tectonic_setting": "subduction",
        }
    )
    return pd.concat([frequent, rare], ignore_index=True)


def test_two_activity_regimes_are_separated(rng):
    p = profiles(rng)
    labels, sil = fit_assign(p, VolcanoClusterParams(n_clusters=2))
    assert labels.iloc[:15].nunique() == 1
    assert labels.iloc[15:].nunique() == 1
    assert labels.iloc[0] != labels.iloc[15]
    assert sil > 0.5

    desc = describe_clusters(p, labels)
    assert set(desc["top_type"]) == {"shield", "stratovolcano"}


def test_missing_values_are_tolerated(rng):
    p = profiles(rng)
    p.loc[0, "max_vei"] = np.nan
    p.loc[1, "volcano_type"] = None
    labels, _ = fit_assign(p, VolcanoClusterParams(n_clusters=2))
    assert labels.notna().all()


def test_params_dict_records_silhouette():
    assert params_dict(VolcanoClusterParams(), 0.7)["silhouette"] == 0.7
