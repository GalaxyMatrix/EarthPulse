import pandas as pd
import pytest

from ml.common.evaluation import (
    MatchParams,
    match_to_events,
    precision_at_k,
    hit_rate,
    random_precision,
    retrieval_report,
    skill_score,
    wet_day_scores,
)
from ml.common.model_card import render


@pytest.fixture
def events():
    return pd.DataFrame({"event_id": ["E1", "E2"], "lat": [38.0, 40.0], "lon": [23.0, 22.0],
                         "first_date": ["2024-07-10", "2024-08-01"], "last_date": ["2024-07-20", "2024-08-05"]})


@pytest.fixture
def candidates():
    return pd.DataFrame(
        {
            "lat":  [38.05, 38.05, 40.0, 10.0],
            "lon":  [23.05, 23.05, 22.0, 10.0],
            "date": ["2024-07-09", "2024-07-30", "2024-08-03", "2024-07-15"],
            "score": [0.9, 0.8, 0.7, 0.95],
        }
    )


def test_matching_respects_space_and_time(candidates, events):
    m = match_to_events(candidates, events, MatchParams(buffer_km=25, before_days=2, after_days=2))
    assert m.loc[0, "matched_event_id"] == "E1"
    assert pd.isna(m.loc[1, "matched_event_id"])  # right place, 10 days after the event ended
    assert m.loc[2, "matched_event_id"] == "E2"
    assert pd.isna(m.loc[3, "matched_event_id"])  # far away
    assert m.loc[0, "lead_days"] == -1.0         # one day before the event started
    assert m.loc[0, "match_dist_km"] < 10


def test_retrieval_metrics(candidates, events):
    m = match_to_events(candidates, events)
    assert precision_at_k(m, 1) == 0.0           # top score is the far-away candidate
    assert precision_at_k(m, 2) == 0.5
    assert random_precision(m) == 0.5
    assert hit_rate(m, events) == 1.0
    assert hit_rate(m, events, k=2) == 0.5
    names = {r["metric"] for r in retrieval_report(m, events, ks=(1, 2))}
    assert {"precision_at_1", "hit_rate_at_2", "median_lead_days", "random_precision"} <= names


def test_skill_scores():
    y = [0, 5, 10]
    assert skill_score(y, y, [5, 5, 5]) == 1.0
    assert skill_score(y, [5, 5, 5], [5, 5, 5]) == 0.0
    s = wet_day_scores([0, 5, 10, 0], [0, 5, 0, 3])
    assert (s["pod"], s["far"], s["csi"]) == (0.5, 0.5, pytest.approx(1 / 3))


def test_model_card_renders():
    md = render(name="fire_anomaly_if", version="1", hazard="wildfire", method="Isolation Forest",
                intended_use="x", not_for="y", data={"source": "FIRMS"}, params={"n": 1},
                evaluations=[{"metric": "precision_at_50", "value": 0.42, "baseline": "random_precision", "details": {"k": 50}}],
                limitations=["z"])
    assert "| precision_at_50 | 0.420 | random_precision |" in md
    assert "exploratory research tool" in md
