# Model card: `fire_anomaly_if` v20261120T185401

- **Hazard:** wildfire
- **Method:** Isolation Forest on FIRMS VIIRS features + H3×month FRP baseline
- **Region:** attica
- **Training window:** 2025-01-01 → 2025-12-31
- **MLflow run:** —
- **Generated:** 2026-11-20 18:58 UTC

## Intended use
Rank wildfire hotspot detections by how unusual they are relative to the same H3 cell and calendar month, to surface candidates for analyst review.

## Not intended for
Operational fire alerts, ignition prediction, burned-area mapping, or comparison of scores with other hazards.

## Data
- **detections:** NASA FIRMS VIIRS S-NPP 375 m
- **reference:** NASA EONET wildfires (evaluation only)
- **features:** frp, brightness, bright_t31, scan, track, cell FRP z/ratio, day/night, confidence
- **region:** attica
- **params_keys:** baseline, features, max_samples, n_estimators, random_state, contamination
- **artifact:** ..\data\processed\models\fire_anomaly_if\20261120T185401\model.joblib
- **data_manifest_hash:** 768ab237a5fb07b5

## Parameters
```json
{
  "baseline": "h3_r5 x month",
  "features": [
    "frp",
    "brightness",
    "bright_t31",
    "scan",
    "track",
    "frp_z_cell",
    "frp_log_ratio_cell",
    "count_ratio_cell",
    "is_day",
    "conf_high"
  ],
  "max_samples": 2048,
  "n_estimators": 300,
  "random_state": 42,
  "contamination": "auto"
}
```

## Evaluation
| metric | value | baseline | details |
|---|---|---|---|
| daily_precision_at_20 | 0.031 | — | {"k": 20, "n_days": 64, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| daily_precision_at_5 | 0.031 | — | {"k": 5, "n_days": 64, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| frac_before_event | 0.000 | — | {"n": 662, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| frp_rank.daily_precision_at_20 | 0.031 | — | {"k": 20, "n_days": 64, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| frp_rank.daily_precision_at_5 | 0.031 | — | {"k": 5, "n_days": 64, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| frp_rank.precision_at_10 | 1.000 | random_precision | {"k": 10, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| frp_rank.precision_at_100 | 0.750 | random_precision | {"k": 100, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| frp_rank.precision_at_50 | 0.740 | random_precision | {"k": 50, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| frp_rank.precision_at_500 | 0.622 | random_precision | {"k": 500, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| hit_rate | 1.000 | — | {"n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| hit_rate_at_10 | 1.000 | — | {"k": 10, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| hit_rate_at_100 | 1.000 | — | {"k": 100, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| hit_rate_at_50 | 1.000 | — | {"k": 50, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| hit_rate_at_500 | 1.000 | — | {"k": 500, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| lift_vs_frp_rank.daily_precision_at_20 | 0.000 | frp_rank | {"k": 20, "n_days": 64, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| lift_vs_frp_rank.daily_precision_at_5 | 0.000 | frp_rank | {"k": 5, "n_days": 64, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| lift_vs_frp_rank.precision_at_10 | -0.100 | frp_rank | {"k": 10, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| lift_vs_frp_rank.precision_at_100 | 0.020 | frp_rank | {"k": 100, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| lift_vs_frp_rank.precision_at_50 | 0.100 | frp_rank | {"k": 50, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| lift_vs_frp_rank.precision_at_500 | 0.022 | frp_rank | {"k": 500, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| median_lead_days | 2.000 | — | {"n": 662, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| precision_at_10 | 0.900 | random_precision | {"k": 10, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| precision_at_100 | 0.770 | random_precision | {"k": 100, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| precision_at_50 | 0.840 | random_precision | {"k": 50, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| precision_at_500 | 0.644 | random_precision | {"k": 500, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |
| random_precision | 0.500 | — | {"n": 1325, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "after_days": 2, "before_days": 2, "n_candidates": 1325} |

## Limitations
- Scores are relative within the scored window, not calibrated probabilities.
- feature_contrib is a standardized deviation view, not SHAP attribution.
- New H3 cells without baseline history get neutral cell features.
- Evaluation against EONET is a coarse spatial-temporal match, not ground truth.

---
_EarthPulse is an exploratory research tool. Outputs are candidates for investigation, not warnings, and scores are not comparable across hazards._
