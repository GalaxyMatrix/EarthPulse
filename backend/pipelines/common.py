"""Shared ingest plumbing: regions config, raw-file archiving, ingest-run bookkeeping, upserts."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.config import settings
from app.db import session_scope
from app.models import IngestRun, Source

MAX_PARAMS = 60_000  # Postgres allows at most 65535 bind parameters per statement


def load_regions() -> dict[str, dict]:
    with (Path(settings.config_dir) / "regions.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)["regions"]


def get_region(key: str) -> dict:
    regions = load_regions()
    if key not in regions:
        raise KeyError(f"Unknown region {key!r}; known: {', '.join(regions)}")
    return regions[key]


def date_chunks(start: date, end: date, days: int) -> Iterator[tuple[date, date]]:
    """[start, end] as consecutive inclusive windows of at most `days` days."""
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=days - 1), end)
        yield cur, stop
        cur = stop + timedelta(days=1)


def point_ewkt(lon: float, lat: float) -> str:
    return f"SRID=4326;POINT({lon} {lat})"


def save_raw(source_key: str, name: str, payload: bytes) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = Path(settings.data_dir) / "raw" / source_key / stamp / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _jsonable(params: dict) -> dict:
    return {k: v.isoformat() if isinstance(v, date) else v for k, v in params.items()}


@contextmanager
def ingest_run(source_key: str, params: dict) -> Iterator[IngestRun]:
    """Yields a committed IngestRun; the caller fills rows_in / rows_written / raw_path.

    The run row lives in its own transactions, so a failed ingest still leaves status='failed' and the error."""
    with session_scope() as session:
        source_id = session.scalar(select(Source.id).where(Source.key == source_key))
        if source_id is None:
            raise ValueError(f"Unknown source {source_key!r}; run `python -m pipelines.cli seed-sources` first")
        run = IngestRun(source_id=source_id, status="running", params=_jsonable(params))
        session.add(run)
    try:
        yield run
    except BaseException as exc:
        _finish(run, status="failed", error=f"{type(exc).__name__}: {exc}"[:4000])
        raise
    _finish(run, status="ok")


def _finish(run: IngestRun, *, status: str, error: str | None = None) -> None:
    run.status = status
    with session_scope() as session:
        session.execute(
            update(IngestRun)
            .where(IngestRun.id == run.id)
            .values(
                status=status, error=error, finished_at=func.now(),
                rows_in=run.rows_in, rows_written=run.rows_written, raw_path=run.raw_path,
            )
        )


def upsert(
    session: Session, model, rows: list[dict], *, conflict: list[str], update_cols: list[str] | None = None
) -> int:
    """INSERT ... ON CONFLICT in parameter-safe chunks. Rows must be unique on `conflict` within the call."""
    if not rows:
        return 0
    cols = update_cols if update_cols is not None else [c for c in rows[0] if c not in conflict]
    step = max(1, MAX_PARAMS // len(rows[0]))
    written = 0
    for i in range(0, len(rows), step):
        stmt = insert(model).values(rows[i : i + step])
        if cols:
            stmt = stmt.on_conflict_do_update(index_elements=conflict, set_={c: stmt.excluded[c] for c in cols})
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=conflict)
        written += session.execute(stmt).rowcount
    return written