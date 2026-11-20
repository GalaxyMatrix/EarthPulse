from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from app.deps import DbSession
from app.geojson import jsonable
from app.models import Evaluation, ModelRun

router = APIRouter()


def _run_dict(run: ModelRun) -> dict:
    return {
        "id": run.id,
        "name": run.name,
        "version": run.version,
        "hazard": run.hazard,
        "region": run.region,
        "params": run.params,
        "train_start": jsonable(run.train_start),
        "train_end": jsonable(run.train_end),
        "artifact_path": run.artifact_path,
        "model_card_path": run.model_card_path,
        "notes": run.notes,
        "registry_alias": run.registry_alias,
        "data_manifest_hash": run.data_manifest_hash,
        "created_at": jsonable(run.created_at),
    }


@router.get("")
def list_model_runs(
    session: DbSession,
    hazard: str | None = Query(None),
    name: str | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    stmt = select(ModelRun).order_by(ModelRun.id.desc()).limit(limit)
    if hazard:
        stmt = stmt.where(ModelRun.hazard == hazard)
    if name:
        stmt = stmt.where(ModelRun.name == name)
    return [_run_dict(r) for r in session.scalars(stmt).all()]


@router.get("/{run_id}")
def get_model_run(run_id: int, session: DbSession):
    run = session.get(ModelRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"model run {run_id} not found")
    return _run_dict(run)


@router.get("/{run_id}/evaluations")
def list_evaluations(run_id: int, session: DbSession):
    run = session.get(ModelRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"model run {run_id} not found")
    rows = session.scalars(
        select(Evaluation).where(Evaluation.model_run_id == run_id).order_by(Evaluation.metric)
    ).all()
    return [
        {
            "id": e.id,
            "model_run_id": e.model_run_id,
            "metric": e.metric,
            "value": e.value,
            "eval_start": jsonable(e.eval_start),
            "eval_end": jsonable(e.eval_end),
            "baseline": e.baseline,
            "details": e.details,
        }
        for e in rows
    ]
