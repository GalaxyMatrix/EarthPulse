from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, timedelta
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from app.models import Event, FireAnomalyScore, FireCluster, FireDetection, ModelRun
from ml.common.evaluation import MatchParams, cluster_coverage_report, match_to_events, precision_at_k, retrieval_report
from pipelines.evaluate import envelope, lift_rows, prefixed, require_out_of_sample, with_context, write_evaluations


EONET_CATEGORY = "wildfires"



@dataclass(frozen=True)
class FireEvalParams:
    match: MatchParams = field(default_factory=lambda: MatchParams(buffer_km=25.0, before_days=2, after_days=2))
    ks: tuple[int, ...] = (10, 50, 100, 500)
    daily_ks: tuple[int, ...] = (5, 20)




def load_events(session: Session, bbox, start: date, end: date, pad_days: int) -> pd.DataFrame:
    lo, hi = start - timedelta(days=pad_days), end + timedelta(days=pad_days)
    stmt = select(
        Event.id.label("event_id"),
        func.ST_Y(Event.geom).label("lat"),
        func.ST_X(Event.geom).label("lon"),
        Event.first_date,
        Event.last_date,

    ).where(
        Event.category == EONET_CATEGORY,
        Event.first_date <= hi,
        Event.last_date >= lo,
        func.ST_Intersects(Event.geom, envelope(bbox)),
    )
    return pd.read_sql(stmt, session.connection())



def load_scored_detections(session: Session, run_id: int, bbox, start: date, end: date) -> pd.DataFrame:
    w, s, e, n = bbox
    stmt = (
        select(
            FireDetection.id.label("detection_id"),
            FireDetection.lat,
            FireDetection.lon,
            FireDetection.acq_date.label("date"),
            FireDetection.frp,
            FireAnomalyScore.score,
        )
        .join(FireAnomalyScore, FireAnomalyScore.detection_id == FireDetection.id)
        .where(
            FireAnomalyScore.model_run_id == run_id,
            FireDetection.acq_date.between(start, end),
            FireDetection.lon.between(w, e),
            FireDetection.lat.between(s, n),
        )

    )
    return pd.read_sql(stmt, session.connection())



def load_clusters(session: Session, run_id: int, bbox, start: date, end: date) -> pd.DataFrame:
    stmt = select(
        FireCluster.id.label("cluster_id"),
        func.ST_Y(FireCluster.centroid).label("lat"),
        func.ST_X(FireCluster.centroid).label("lon"),
        FireCluster.cluster_date.label("date"),
        FireCluster.n_detections,
        FireCluster.frp_sum,
    ).where(
        FireCluster.model_run_id == run_id,
        FireCluster.cluster_date.between(start, end),
        func.ST_Intersects(FireCluster.centroid, envelope(bbox)),
    )
    return pd.read_sql(stmt, session.connection())



def daily_precision(matched: pd.DataFrame, ks, score_col: str = "score") -> list[dict]:
    rows = []
    for k in ks:
        per_day = [p for _, g in matched.groupby("date") if (p := precision_at_k(g, k, score_col)) == p]
        rows.append({
            "metric": f"daily_precision_at_{k}",
            "value": sum(per_day) / len(per_day) if per_day else float("nan"),
            "baseline": None,
            "details": {"k": k, "n_days": len(per_day)},

        })
    return rows 


def _precision_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if "precision_at" in r["metric"]] 



def compute_anomaly_eval(candidates: pd.DataFrame, events: pd.DataFrame, params: FireEvalParams = FireEvalParams()) -> list[dict]:
    matched = match_to_events(candidates, events, params.match)
    model = retrieval_report(matched, events, params.ks) + daily_precision(matched, params.daily_ks)
    frp = retrieval_report(matched, events, params.ks, score_col="frp") + daily_precision(matched, params.daily_ks, "frp")
    return (
        model
        + prefixed(_precision_rows(frp), "frp_rank")
        + lift_rows(_precision_rows(model), _precision_rows(frp), "frp_rank")
    )



def compute_cluster_eval(clusters: pd.DataFrame, events: pd.DataFrame, params: FireEvalParams = FireEvalParams()) -> list[dict]:
    """clusters: lat, lon (centroid), date, n_detections."""
    matched = match_to_events(clusters, events, params.match)
    rows = cluster_coverage_report(matched, events)
    dist = matched["match_dist_km"].dropna()
    if len(dist):
        rows.append({"metric": "median_match_dist_km", "value": float(dist.median()), "baseline": None, "details": {"n": int(len(dist))}})
    return rows

    
# --- runners ---------------------------------------------------------------
def run_anomaly_eval(session: Session, *, run_id: int, bbox, start: date, end: date, params: FireEvalParams = FireEvalParams()) -> int:
    run = session.get(ModelRun, run_id)
    require_out_of_sample(run, start)
    candidates = load_scored_detections(session, run_id, bbox, start, end)
    if candidates.empty:
        raise RuntimeError(f"No scored detections for run {run_id} in {start}..{end}; run score/fire_anomaly first.")
    events = load_events(session, bbox, start, end, max(params.match.before_days, params.match.after_days))
    rows = with_context(
        compute_anomaly_eval(candidates, events, params),
        buffer_km=params.match.buffer_km, before_days=params.match.before_days, after_days=params.match.after_days,
        n_candidates=int(len(candidates)), n_events=int(len(events)), reference="eonet:wildfires",
    )
    return write_evaluations(session, run, rows, eval_start=start, eval_end=end)


def run_cluster_eval(session: Session, *, run_id: int, bbox, start: date, end: date, params: FireEvalParams = FireEvalParams()) -> int:
    run = session.get(ModelRun, run_id)  # DBSCAN has no training window, so no out-of-sample check
    clusters = load_clusters(session, run_id, bbox, start, end)
    if clusters.empty:
        raise RuntimeError(f"No clusters for run {run_id} in {start}..{end}; run score/fire_clusters first.")
    events = load_events(session, bbox, start, end, max(params.match.before_days, params.match.after_days))
    rows = with_context(
        compute_cluster_eval(clusters, events, params),
        buffer_km=params.match.buffer_km, n_clusters=int(len(clusters)), n_events=int(len(events)), reference="eonet:wildfires",
    )
    return write_evaluations(session, run, rows, eval_start=start, eval_end=end)



