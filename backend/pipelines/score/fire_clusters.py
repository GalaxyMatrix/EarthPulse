"""Daily DBSCAN clustering of FIRMS detections; one ModelRun per scoring window."""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from app.db import session_scope
from app.models import FireCluster, FireClusterMember
from ml.wildfire.clusters import ClusterParams, params_dict, summarize_clusters
from pipelines.common import get_region
from pipelines.features.fire_features import load_detections
from pipelines.runs import create_run

log = logging.getLogger(__name__)

NAME = "fire_clusters_dbscan"
HAZARD = "wildfire"


def score(
    region_key: str,
    start: date,
    end: date,
    *,
    params: ClusterParams = ClusterParams(),
) -> int:
    bbox = get_region(region_key)["bbox"]
    with session_scope() as session:
        detections = load_detections(session, start=start, end=end, bbox=bbox)
        if detections.empty:
            raise RuntimeError(f"No detections to cluster in {start}..{end}")

        run = create_run(
            session,
            name=NAME,
            hazard=HAZARD,
            region=region_key,
            params=params_dict(params),
            train_start=start,
            train_end=end,
            notes="DBSCAN (haversine) daily clusters; no supervised training window",
            registry_alias="challenger",
        )

        n_clusters = 0
        for day, day_df in detections.groupby(pd.to_datetime(detections["acq_date"]).dt.date):
            summary = summarize_clusters(day_df, params)
            if summary.empty:
                continue
            for row in summary.itertuples(index=False):
                cluster = FireCluster(
                    model_run_id=run.id,
                    cluster_date=day,
                    cluster_label=int(row.cluster_label),
                    hull=f"SRID=4326;{row.hull_wkt}",
                    centroid=f"SRID=4326;{row.centroid_wkt}",
                    n_detections=int(row.n_detections),
                    frp_sum=row.frp_sum,
                    frp_max=row.frp_max,
                    area_km2=row.area_km2,
                )
                session.add(cluster)
                session.flush()
                for det_idx in row.member_index:
                    det_id = int(day_df.loc[det_idx, "detection_id"])
                    session.add(FireClusterMember(cluster_id=cluster.id, detection_id=det_id))
                n_clusters += 1

        log.info("wrote %d clusters for run %s", n_clusters, run.id)
        return run.id
