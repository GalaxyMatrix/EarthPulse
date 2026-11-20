from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, timedelta
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from app.models import Event, ModelRun, RainAnomaly, RainCell
from ml.common.evaluation import MatchParams, match_to_events, retrieval_report
from pipelines.evaluate import envelope, lift_rows, prefixed, require_out_of_sample, with_context, write_evaluations
from pipelines.evaluate.fire_vs_eonet import _precision_rows, daily_precision



EONET_CATEGORY = "floods"


@dataclass(frozen=True)
class RainEvalParams:
    match: MatchParams = field(default_factory=lambda: MatchParams(buffer_km=75.0, before_days=5, after_days=1))
    ks: tuple[int, ...] = (10, 50, 100)
    daily_ks: tuple[int, ...] = (5,)
    window_days: int | None = None   # z-score runs store 1/3/7-day rows; pick one. Isolation Forest runs store one window.
    min_score: float | None = None   # optional pre-filter to keep the candidate table small




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



def load_anomalies(session: Session, run_id: int, bbox, start: date, end: date, params: RainEvalParams) -> list[dict]:
    stmt = (
        select(
            RainAnomaly.cell_id,
            func.ST_Y(RainCell.centroid).label("lat"),
            func.ST_X(RainCell.centroid).label("lon"),
            RainAnomaly.obs_date.label("date"),
            RainAnomaly.precip_mm,
            RainAnomaly.z,
            RainAnomaly.score,
        )
        .join(RainCell, RainCell.id == RainAnomaly.cell_id)
        .where(
            RainAnomaly.model_run_id == run_id,
            RainAnomaly.obs_date.between(start, end),
            func.ST_Intersects(RainCell.centroid, envelope(bbox)),
        )
    )
    if params.window_days is not None:
        stmt = stmt.where(RainAnomaly.window_days == params.window_days)
    if params.min_score is not None:
        stmt = stmt.where(RainAnomaly.score >= params.min_score)
    return pd.read_sql(stmt, session.connection())



def compute_eval(candidates: pd.DataFrame, events: pd.DataFrame, params: RainEvalParams = RainEvalParams()) -> list[dict]:
    """candidates: lat, lon, date, score, precip_mm."""
    matched = match_to_events(candidates, events, params.match)
    model = retrieval_report(matched, events, params.ks) + daily_precision(matched, params.daily_ks)
    precip = retrieval_report(matched, events, params.ks, score_col="precip_mm") + daily_precision(matched, params.daily_ks, "precip_mm")
    return (
        model
        + prefixed(_precision_rows(precip), "precip_rank")
        + lift_rows(_precision_rows(model), _precision_rows(precip), "precip_rank")
    )


    
def run_eval(session: Session, *, run_id: int, bbox, start: date, end: date, params: RainEvalParams = RainEvalParams()) -> int:
    run = session.get(ModelRun, run_id)
    require_out_of_sample(run, start)  # set train_end to the climatology window's end when creating anomaly runs
    candidates = load_anomalies(session, run_id, bbox, start, end, params)
    if candidates.empty:
        raise RuntimeError(f"No rain anomalies for run {run_id} in {start}..{end}; run score/rain_anomaly first.")
    events = load_events(session, bbox, start, end, max(params.match.before_days, params.match.after_days))
    rows = with_context(
        compute_eval(candidates, events, params),
        buffer_km=params.match.buffer_km, before_days=params.match.before_days, after_days=params.match.after_days,
        window_days=params.window_days, n_candidates=int(len(candidates)), n_events=int(len(events)), reference="eonet:floods",
    )
    return write_evaluations(session, run, rows, eval_start=start, eval_end=end)
