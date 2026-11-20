# Model card: `rain_climatology` v20261120T185652

- **Hazard:** rainfall
- **Method:** Per-cell day-of-year window mean/std and quantiles
- **Region:** attica
- **Training window:** 2023-01-01 → 2023-10-31
- **MLflow run:** —
- **Generated:** 2026-11-20 18:58 UTC

## Intended use
Provide a seasonal baseline for rainfall anomaly and forecast features.

## Not intended for
Direct flood warnings or climate-change trend analysis.

## Data
- **precip:** GPM IMERG Final daily (regional cells)
- **region:** attica
- **params_keys:** min_years, window_days
- **artifact:** —
- **data_manifest_hash:** 71177a023b8b12e9

## Parameters
```json
{
  "min_years": 1,
  "window_days": 31
}
```

## Evaluation
_No evaluation recorded yet._

## Limitations
- Needs multiple years per cell for stable quantiles (min_years).
- Window wraps at year-end; leap days are included in DOY 366.

---
_EarthPulse is an exploratory research tool. Outputs are candidates for investigation, not warnings, and scores are not comparable across hazards._
