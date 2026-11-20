from __future__ import annotations
import math
from datetime import date
from sqlalchemy import delete, func
from sqlalchemy.orm import Session
from app.config import settings
from app.models import Evaluation, ModelRun



def envelope(bbox: tuple[float, float, float, float]):
    w, s, e, n = bbox
    return func.ST_MakeEnvelope(w, s, e, n, 4326)


def require_out_of_sample(run: ModelRun, eval_start: date) -> None:
    if run.train_end is not None and eval_start <= run.train_end:
        raise ValueError(
            f"{run.name} v{run.version} was trained up to {run.train_end};"
            f"evaluation must start after that (got {eval_start})."
        )


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)



def prefixed(rows: list[dict], prefix: str) -> list[dict]:
    """Rename rows that describe a baseline, e.g. precision_at_50 -> frp_rank.precision_at_50."""
    return [r | {"metric": f"{prefix}.{r['metric']}"} for r in rows]


    
def lift_rows(model_rows: list[dict], baseline_rows: list[dict], baseline_name: str) -> list[dict]:
    """model - baseline for every metric both have. > 0 means the model adds something."""
    base = {r["metric"]: r["value"] for r in baseline_rows}
    out = []
    for r in model_rows:
        b = base.get(r["metric"])
        if _finite(r["value"]) and _finite(b):
            out.append({"metric": f"lift_vs_{baseline_name}.{r['metric']}", "value": r["value"] - b,
                        "baseline": baseline_name, "details": r.get("details") or {}})
    return out



def with_context(rows: list[dict], **context) -> list[dict]:
    return [r | {"details": (r.get("details") or {}) | context} for r in rows]




def write_evaluations(session: Session, run: ModelRun, rows: list[dict], *, eval_start: date, eval_end: date) -> int:
    """Replace this run's metrics for the same window, then mirror them to MLflow."""
    rows = [r for r in rows if _finite(r.get("value"))]
    session.execute(
        delete(Evaluation).where(
            Evaluation.model_run_id == run.id,
            Evaluation.eval_start == eval_start,
            Evaluation.eval_end == eval_end,
            Evaluation.metric.in_([r["metric"] for r in rows]),
        )
    )
    session.add_all(
        Evaluation(
            model_run_id=run.id, metric=r["metric"], value=float(r["value"]), baseline=r.get("baseline"),
            details=r.get("details") or None, eval_start=eval_start, eval_end=eval_end,
        )
        for r in rows
    )
    session.commit()
    _log_to_mlflow(run, rows)
    return len(rows)



def _log_to_mlflow(run: ModelRun, rows: list[dict]) -> None:
    if not (run.mlflow_run_id and settings.mlflow_tracking_uri):
        return
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    for r in rows:
        client.log_metric(run.mlflow_run_id, r["metric"], float(r["value"]))



