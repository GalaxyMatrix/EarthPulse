"""Score FIRMS detections with a trained fire_anomaly_if run."""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from app.db import session_scope
from app.models import FireAnomalyScore, ModelRun
from ml.wildfire.anomaly import FireAnomalyParams, make_features, score as score_model
from pipelines.common import get_region, upsert
from pipelines.features import to_records
from pipelines.features.fire_features import load_detections
from pipelines.runs import load_artifact

log = logging.getLogger(__name__)

COLUMNS = ["detection_id", "model_run_id", "score", "percentile", "rank_in_day", "feature_contrib"]


def score(run_id: int, start: date, end: date) -> int:
    with session_scope() as session:
        run = session.get(ModelRun, run_id)
        if run is None or not run.artifact_path:
            raise ValueError(f"No fire anomaly run {run_id} with an artifact")
        if run.region is None:
            raise ValueError(f"Run {run_id} has no region")
        art = load_artifact(run.artifact_path)
        model, baseline = art["model"], art["baseline"]
        raw = {**(run.params or {}), **(art.get("params") or {})}
        fields = FireAnomalyParams.__dataclass_fields__
        kwargs = {k: raw[k] for k in fields if k in raw and k != "features"}
        if "features" in raw:
            kwargs["features"] = tuple(raw["features"])
        params = FireAnomalyParams(**kwargs)

        bbox = get_region(run.region)["bbox"]
        detections = load_detections(session, start=start, end=end, bbox=bbox)
        if detections.empty:
            raise RuntimeError(f"No detections to score in {start}..{end}")
        features = make_features(detections, baseline)
        out = score_model(model, features, params)
        out["detection_id"] = features["detection_id"].to_numpy()
        out["model_run_id"] = run_id
        # ml.wildfire ranks globally; store true per-day ranks for the UI / eval
        out["rank_in_day"] = (
            out.groupby(pd.to_datetime(features["acq_date"]).dt.date)["score"]
            .rank(ascending=False, method="first")
            .astype(int)
        )
        n = upsert(
            session,
            FireAnomalyScore,
            to_records(out, COLUMNS),
            conflict=["detection_id", "model_run_id"],
        )
        log.info("scored %d detections for run %s", len(out), run_id)
        return len(out) if n < 0 else n
