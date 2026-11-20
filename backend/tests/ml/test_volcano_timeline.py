import pandas as pd
import pytest

from ml.volcano.timeline import repose_percentile, volcano_profile, volcano_timeline


def eruptions():
    return pd.DataFrame(
        {
            "start_date": ["2000-01-01", "2002-01-01", "2005-01-01", "2010-01-01"],
            "end_date": ["2000-01-31", "2002-03-02", "2005-01-11", "2010-04-10"],
            "vei": [2, 3, None, 1],
        }
    )


def test_timeline_durations_and_repose():
    t = volcano_timeline(eruptions())
    assert t["duration_days"].tolist() == [30, 60, 10, 99]
    assert t["repose_days"].iloc[1] == (pd.Timestamp("2002-01-01") - pd.Timestamp("2000-01-31")).days


def test_profile_of_quiet_volcano():
    p = volcano_profile(eruptions(), "2020-01-01")
    assert p["n_events"] == 4
    assert p["is_ongoing"] is False
    assert p["is_active"] is False
    assert p["days_since_last"] == (pd.Timestamp("2020-01-01") - pd.Timestamp("2010-04-10")).days
    assert p["max_vei"] == 3.0
    assert p["events_per_decade"] == pytest.approx(4.0, abs=0.01)


def test_ongoing_eruption_is_active():
    e = pd.concat([eruptions(), pd.DataFrame({"start_date": ["2019-06-01"], "end_date": [None], "vei": [2]})])
    p = volcano_profile(e, "2020-01-01")
    assert p["is_ongoing"] and p["is_active"]
    assert p["days_since_last"] is None
    assert repose_percentile(e, "2020-01-01") is None


def test_repose_percentile():
    # current repose (~3,550 days) is longer than every past repose
    assert repose_percentile(eruptions(), "2020-01-01") == 100.0
    # one day after the last eruption: shorter than all past reposes
    assert repose_percentile(eruptions(), "2010-04-11") == 0.0


def test_too_few_events_gives_no_stats():
    p = volcano_profile(eruptions().head(2), "2020-01-01")
    assert p["median_repose_days"] is None and p["events_per_decade"] is None
