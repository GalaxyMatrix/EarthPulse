import pandas as pd

from ml.wildfire.clusters import ClusterParams, cluster_detections, eps_sensitivity, summarize_clusters


def two_fires_and_noise():
    return pd.DataFrame(
        {
            # fire A near Athens, fire B ~100 km away, one isolated hotspot
            "lat": [38.000, 38.005, 38.010, 38.004, 39.000, 39.004, 39.008, 40.500],
            "lon": [23.000, 23.004, 23.002, 23.008, 22.000, 22.003, 22.006, 25.000],
            "frp": [10.0, 20.0, 30.0, 40.0, 5.0, 6.0, 7.0, 99.0],
        }
    )


def test_two_clusters_and_noise():
    labels = cluster_detections(two_fires_and_noise())
    assert labels[labels >= 0].nunique() == 2
    assert labels.iloc[-1] == -1
    assert labels.iloc[0] == labels.iloc[3]
    assert labels.iloc[0] != labels.iloc[4]


def test_empty_input():
    assert cluster_detections(pd.DataFrame(columns=["lat", "lon"])).empty


def test_summary_stats():
    summary = summarize_clusters(two_fires_and_noise()).sort_values("n_detections", ascending=False)
    assert summary["n_detections"].tolist() == [4, 3]
    assert summary.iloc[0]["frp_sum"] == 100.0
    assert summary.iloc[0]["frp_max"] == 40.0
    assert (summary["area_km2"] > 0).all()
    assert summary.iloc[0]["hull_wkt"].startswith("POLYGON")


def test_larger_eps_merges_clusters():
    sens = eps_sensitivity(two_fires_and_noise(), eps_values=(2.5, 200.0)).set_index("eps_km")
    assert sens.loc[2.5, "n_clusters"] == 2
    assert sens.loc[200.0, "n_clusters"] == 1


def test_min_samples_turns_small_cluster_into_noise():
    labels = cluster_detections(two_fires_and_noise(), ClusterParams(min_samples=4))
    assert labels[labels >= 0].nunique() == 1
