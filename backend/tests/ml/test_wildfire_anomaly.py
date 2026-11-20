import pandas as pd

from ml.wildfire.anomaly import FireAnomalyParams, build_baseline, fit, make_features, params_dict, score

PARAMS = FireAnomalyParams(n_estimators=100)


def test_baseline_has_one_row_per_cell_month(fire_history):
    base = build_baseline(fire_history)
    assert not base.duplicated(["h3_r5", "month"]).any()
    assert len(base) == 3 * 12
    assert (base["frp_std"] > 0).all()
    assert {"frp_p50", "frp_mean", "count_per_day"} <= set(base.columns)


def test_unknown_cell_gets_neutral_features(fire_history):
    base = build_baseline(fire_history)
    new = fire_history.head(1).assign(h3_r5="never_seen")
    feats = make_features(new, base)
    assert feats["frp_z_cell"].iloc[0] == 0.0
    assert feats["count_ratio_cell"].iloc[0] == 1.0


def test_extreme_detection_ranks_first(fire_history):
    base = build_baseline(fire_history)
    model = fit(make_features(fire_history, base), PARAMS)

    day = fire_history.sample(200, random_state=0).assign(acq_date=pd.Timestamp("2025-07-15")).reset_index(drop=True)
    day.loc[0, ["frp", "brightness", "bright_t31"]] = [2500.0, 420.0, 330.0]
    scored = score(model, make_features(day, base), PARAMS)

    assert scored["rank_in_day"].iloc[0] == 1
    assert scored["percentile"].iloc[0] == 100.0
    assert scored["rank_in_day"].is_unique
    contrib = scored["feature_contrib"].iloc[0]
    assert contrib["frp"]["value"] == 2500.0
    assert contrib["frp"]["z"] > 3


def test_params_dict_is_serialisable():
    import json

    json.dumps(params_dict(PARAMS))
