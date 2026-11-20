# Model card: `rain_anomaly_z` v20261120T185716

- **Hazard:** rainfall
- **Method:** Rolling precip totals vs DOY climatology (z-score / percentile)
- **Region:** attica
- **Training window:** 2023-01-01 → 2023-10-31
- **MLflow run:** —
- **Generated:** 2026-11-20 18:58 UTC

## Intended use
Flag unusually wet cell-days relative to the local seasonal baseline for flood candidate review.

## Not intended for
Flash-flood warnings, urban drainage design, or drought monitoring.

## Data
- **precip:** GPM IMERG Final daily
- **baseline:** rain_climatology ModelRun
- **reference:** NASA EONET floods (evaluation only)
- **region:** attica
- **params_keys:** method, anomaly, window_days, climatology_run_id
- **artifact:** —
- **data_manifest_hash:** —

## Parameters
```json
{
  "method": "z_vs_doy_climatology",
  "anomaly": {
    "windows": [
      1,
      3,
      7
    ],
    "std_floor_mm": 1.0
  },
  "window_days": 7,
  "climatology_run_id": 4
}
```

## Evaluation
| metric | value | baseline | details |
|---|---|---|---|
| daily_precision_at_5 | 0.000 | — | {"k": 5, "n_days": 55, "n_events": 0, "buffer_km": 75.0, "reference": "eonet:floods", "after_days": 1, "before_days": 5, "window_days": 7, "n_candidates": 3850} |
| lift_vs_precip_rank.daily_precision_at_5 | 0.000 | precip_rank | {"k": 5, "n_days": 55, "n_events": 0, "buffer_km": 75.0, "reference": "eonet:floods", "after_days": 1, "before_days": 5, "window_days": 7, "n_candidates": 3850} |
| lift_vs_precip_rank.precision_at_10 | 0.000 | precip_rank | {"k": 10, "n_events": 0, "buffer_km": 75.0, "reference": "eonet:floods", "after_days": 1, "before_days": 5, "window_days": 7, "n_candidates": 3850} |
| lift_vs_precip_rank.precision_at_100 | 0.000 | precip_rank | {"k": 100, "n_events": 0, "buffer_km": 75.0, "reference": "eonet:floods", "after_days": 1, "before_days": 5, "window_days": 7, "n_candidates": 3850} |
| lift_vs_precip_rank.precision_at_50 | 0.000 | precip_rank | {"k": 50, "n_events": 0, "buffer_km": 75.0, "reference": "eonet:floods", "after_days": 1, "before_days": 5, "window_days": 7, "n_candidates": 3850} |
| precip_rank.daily_precision_at_5 | 0.000 | — | {"k": 5, "n_days": 55, "n_events": 0, "buffer_km": 75.0, "reference": "eonet:floods", "after_days": 1, "before_days": 5, "window_days": 7, "n_candidates": 3850} |
| precip_rank.precision_at_10 | 0.000 | random_precision | {"k": 10, "n_events": 0, "buffer_km": 75.0, "reference": "eonet:floods", "after_days": 1, "before_days": 5, "window_days": 7, "n_candidates": 3850} |
| precip_rank.precision_at_100 | 0.000 | random_precision | {"k": 100, "n_events": 0, "buffer_km": 75.0, "reference": "eonet:floods", "after_days": 1, "before_days": 5, "window_days": 7, "n_candidates": 3850} |
| precip_rank.precision_at_50 | 0.000 | random_precision | {"k": 50, "n_events": 0, "buffer_km": 75.0, "reference": "eonet:floods", "after_days": 1, "before_days": 5, "window_days": 7, "n_candidates": 3850} |
| precision_at_10 | 0.000 | random_precision | {"k": 10, "n_events": 0, "buffer_km": 75.0, "reference": "eonet:floods", "after_days": 1, "before_days": 5, "window_days": 7, "n_candidates": 3850} |
| precision_at_100 | 0.000 | random_precision | {"k": 100, "n_events": 0, "buffer_km": 75.0, "reference": "eonet:floods", "after_days": 1, "before_days": 5, "window_days": 7, "n_candidates": 3850} |
| precision_at_50 | 0.000 | random_precision | {"k": 50, "n_events": 0, "buffer_km": 75.0, "reference": "eonet:floods", "after_days": 1, "before_days": 5, "window_days": 7, "n_candidates": 3850} |
| random_precision | 0.000 | — | {"n": 3850, "n_events": 0, "buffer_km": 75.0, "reference": "eonet:floods", "after_days": 1, "before_days": 5, "window_days": 7, "n_candidates": 3850} |

## Limitations
- Only one window_days is stored per run (DB unique key).
- IMERG Final lags real time by months.
- EONET floods are sparse and spatially coarse versus 0.1° cells.

---
_EarthPulse is an exploratory research tool. Outputs are candidates for investigation, not warnings, and scores are not comparable across hazards._
