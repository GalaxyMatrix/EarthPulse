"""Train wildfire Isolation Forest; persist model + H3×month baseline."""
from __future__ import annotations

import logging
from datetime import date

from app.db import session_scope
from ml.wildfire.anomaly import FireAnomalyParams, fit, params_dict
from pipelines.common import get_region
from pipelines.features import write_snapshot
from pipelines.features.fire_features import load_training_table
from pipelines.runs import create_run, save_artifact

log = logging.getLogger(__name__)

NAME = "fire_anomaly_if"
HAZARD = "wildfire"


def train(
    region_key: str,
    start: date,
    end: date,
    *,
    params: FireAnomalyParams = FireAnomalyParams(),
) -> int:
    bbox = get_region(region_key)["bbox"]
    with session_scope() as session:
        features, baseline = load_training_table(session, start=start, end=end, bbox=bbox)
        if features.empty:
            raise RuntimeError(f"No VIIRS detections in {start}..{end} for {region_key}")
        if baseline.empty:
            raise RuntimeError(f"Empty fire baseline before {start}; ingest more FIRMS history")

        _, digest = write_snapshot(
            features[["detection_id", *params.features]].reset_index(drop=True), NAME
        )
        model = fit(features, params)
        run = create_run(
            session,
            name=NAME,
            hazard=HAZARD,
            region=region_key,
            params=params_dict(params),
            train_start=start,
            train_end=end,
            data_manifest_hash=digest,
            notes="Isolation Forest on FIRMS VIIRS with H3-r5 × month FRP baseline",
            registry_alias="challenger",
        )
        run.artifact_path = save_artifact(
            NAME, run.version, {"model": model, "baseline": baseline, "params": params_dict(params)}
        )
        log.info("trained %s v%s on %d rows; run_id=%s", NAME, run.version, len(features), run.id)
        return run.id
