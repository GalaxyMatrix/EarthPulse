# Model card: `fire_clusters_dbscan` v20261120T185439

- **Hazard:** wildfire
- **Method:** Daily DBSCAN (haversine) on FIRMS detection coordinates
- **Region:** attica
- **Training window:** 2026-07-01 → 2026-09-30
- **MLflow run:** —
- **Generated:** 2026-11-20 18:58 UTC

## Intended use
Group same-day hotspots into spatial clusters for map display and comparison with catalogue wildfire events.

## Not intended for
Perimeter mapping, fire-spread modelling, or counting distinct ignitions.

## Data
- **detections:** NASA FIRMS VIIRS S-NPP 375 m
- **reference:** NASA EONET wildfires (evaluation only)
- **region:** attica
- **params_keys:** eps_km, metric, algorithm, min_samples, hull_buffer_km
- **artifact:** —
- **data_manifest_hash:** —

## Parameters
```json
{
  "eps_km": 2.5,
  "metric": "haversine",
  "algorithm": "ball_tree",
  "min_samples": 3,
  "hull_buffer_km": 0.4
}
```

## Evaluation
| metric | value | baseline | details |
|---|---|---|---|
| cluster_match_rate | 0.273 | — | {"n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "n_clusters": 22} |
| event_coverage | 1.000 | — | {"n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "n_clusters": 22} |
| median_match_dist_km | 6.183 | — | {"n": 6, "n_events": 1, "buffer_km": 25.0, "reference": "eonet:wildfires", "n_clusters": 22} |

## Limitations
- eps_km and min_samples are heuristic; noise points are dropped.
- Hulls are convex buffers, not burned area.
- Clustering is day-scoped; multi-day fires appear as separate daily clusters.

---
_EarthPulse is an exploratory research tool. Outputs are candidates for investigation, not warnings, and scores are not comparable across hazards._
