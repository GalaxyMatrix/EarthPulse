"""ModelRun bookkeeping: versioning, artifact paths, and inserting run rows."""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import joblib
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import ModelRun


def next_version(session: Session, name: str) -> str:
    """UTC timestamp version; unique with name under uq_model_runs_name_version."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    existing = set(session.scalars(select(ModelRun.version).where(ModelRun.name == name)))
    if stamp not in existing:
        return stamp
    return f"{stamp}_{len(existing)}"


def artifact_dir(name: str, version: str) -> Path:
    path = Path(settings.data_dir) / "processed" / "models" / name / version
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_artifact(name: str, version: str, payload: dict, filename: str = "model.joblib") -> str:
    path = artifact_dir(name, version) / filename
    joblib.dump(payload, path)
    return str(path)


def load_artifact(path: str):
    return joblib.load(path)


def create_run(
    session: Session,
    *,
    name: str,
    hazard: str,
    params: dict,
    region: str | None = None,
    train_start: date | None = None,
    train_end: date | None = None,
    artifact_path: str | None = None,
    data_manifest_hash: str | None = None,
    notes: str | None = None,
    registry_alias: str | None = None,
) -> ModelRun:
    run = ModelRun(
        name=name,
        version=next_version(session, name),
        hazard=hazard,
        region=region,
        params=params,
        train_start=train_start,
        train_end=train_end,
        artifact_path=artifact_path,
        data_manifest_hash=data_manifest_hash,
        notes=notes,
        registry_alias=registry_alias,
    )
    session.add(run)
    session.flush()
    return run
