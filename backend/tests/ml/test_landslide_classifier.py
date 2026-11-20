import numpy as np
import pandas as pd

from ml.landslide.classifier import LandslideClassifierParams, cross_validate, fit, predict, prepare_labels
from ml.landslide.clustering import cluster_events, summarize
from ml.landslide.features import NUMERIC_FEATURES

FAST = LandslideClassifierParams(n_splits=4, min_class_support=10, hgb={"max_iter": 50, "class_weight": "balanced", "random_state": 0})


def labelled_events(rng, n=240):
    """Rain-triggered events have heavy antecedent rain; earthquake-triggered ones don't."""
    trigger = rng.choice(["Downpour", "Earthquake", "Continuous rain"], n, p=[0.45, 0.35, 0.2])
    rain = np.isin(trigger, ["Downpour", "Continuous rain"])
    df = pd.DataFrame({"event_id": range(n), "lat": rng.uniform(-30, 50, n), "lon": rng.uniform(-120, 140, n), "trigger": trigger})
    for col in NUMERIC_FEATURES:
        df[col] = rng.normal(0, 1, n)
    df["rain_7d_mm"] = np.where(rain, rng.uniform(80, 200, n), rng.uniform(0, 20, n))
    df["api"] = np.where(rain, rng.uniform(100, 300, n), rng.uniform(0, 30, n))
    df["land_cover"] = rng.choice(["forest", "cropland", "urban"], n)
    return df


def test_prepare_labels_maps_and_collapses(rng):
    df = labelled_events(rng)
    df.loc[:2, "trigger"] = ["unknown", "Mining", "Volcano"]
    d = prepare_labels(df, FAST)
    assert 0 not in d.index  # 'unknown' dropped
    assert set(d["trigger_class"]) == {"rain", "earthquake", "other"}  # rare classes -> other


def test_spatial_cv_beats_majority_baseline(rng):
    d = prepare_labels(labelled_events(rng), FAST)
    cv = cross_validate(d, FAST)
    assert cv["n_splits"] == 4
    assert cv["macro_f1_mean"] > cv["macro_f1_majority_baseline"] + 0.3
    assert cv["accuracy"] > cv["accuracy_majority_baseline"]
    assert len(cv["confusion_matrix"]) == len(cv["classes"])


def test_predict_returns_probabilities(rng):
    d = prepare_labels(labelled_events(rng), FAST)
    out = predict(fit(d, FAST), d.head(20))
    prob_cols = [c for c in out.columns if c.startswith("p_")]
    assert np.allclose(out[prob_cols].sum(axis=1), 1.0)
    assert out["confidence"].between(0, 1).all()


def test_landslide_swarm_clustering():
    swarm = pd.DataFrame({"lat": [27.70, 27.75, 27.80, 27.72], "lon": [85.30, 85.35, 85.32, 85.40],
                          "event_date": pd.to_datetime(["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-02"]), "trigger": "downpour"})
    same_place_later = swarm.assign(event_date=swarm["event_date"] + pd.Timedelta(days=200))
    lone = pd.DataFrame({"lat": [10.0], "lon": [-75.0], "event_date": pd.to_datetime(["2024-07-02"]), "trigger": ["earthquake"]})
    events = pd.concat([swarm, same_place_later, lone], ignore_index=True)

    labels = cluster_events(events)
    assert labels.iloc[-1] == -1
    assert labels[labels >= 0].nunique() == 2  # same place, 200 days apart -> separate swarms
    s = summarize(events, labels)
    assert s["n_events"].tolist() == [4, 4]
    assert (s["top_trigger"] == "downpour").all()
