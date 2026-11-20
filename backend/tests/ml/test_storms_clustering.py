import pytest

from ml.storms.clustering import StormClusterParams, assign, describe_clusters, fit, nearest_storms, storm_feature_table
from tests.conftest import make_track


def two_storm_families(rng):
    tracks = {}
    for i in range(8):  # long, intense, north-west movers
        tracks[f"nw_{i}"] = make_track(rng, 15 + rng.normal(0, 1), -45 + rng.normal(0, 1), 0.6, -0.8, n=20, base_kts=60, growth_kts=6)
    for i in range(8):  # short, weak, east movers
        tracks[f"e_{i}"] = make_track(rng, 30 + rng.normal(0, 1), -70 + rng.normal(0, 1), 0.05, 0.4, n=6, base_kts=30, growth_kts=1)
    return tracks


def test_feature_table_skips_short_tracks(rng):
    tracks = two_storm_families(rng)
    tracks["too_short"] = make_track(rng, 10, -40, 0.1, 0.1, n=2)
    table = storm_feature_table(tracks)
    assert len(table) == 16
    assert "too_short" not in set(table["event_id"])


def test_families_are_separated(rng):
    table = storm_feature_table(two_storm_families(rng))
    model = fit(table, StormClusterParams(k_range=(2, 3)))
    labels = assign(model, table)
    family = table["event_id"].str.split("_").str[0]
    # no cluster mixes the two families (a family may still be split into sub-groups)
    assert (table.assign(cluster=labels).groupby("cluster")["event_id"].apply(lambda s: s.str.split("_").str[0].nunique()) == 1).all()
    assert set(labels[family == "nw"]).isdisjoint(labels[family == "e"])
    assert model.silhouette > 0.5

    desc = describe_clusters(table, labels)
    assert desc["n"].sum() == 16


def test_nearest_storms_are_same_family(rng):
    table = storm_feature_table(two_storm_families(rng))
    model = fit(table, StormClusterParams(k_range=(2,)))
    near = nearest_storms(model, table, "nw_0", n=3)
    assert near["event_id"].str.startswith("nw_").all()


def test_too_few_storms_raises(rng):
    table = storm_feature_table({f"s{i}": make_track(rng, 15, -50, 0.3, -0.5) for i in range(2)})
    with pytest.raises(ValueError):
        fit(table, StormClusterParams(k_range=(3,)))
