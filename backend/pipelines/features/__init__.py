"""Feature builders: load from PostGIS, call the pure functions in ml/, return or persist feature tables."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from app.config import settings


def to_records(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    """Rows as plain dicts with NaN/NaT replaced by None, ready for an INSERT."""
    sub = df[columns].astype(object)
    return sub.where(sub.notna(), None).to_dict("records")


def table_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return h.hexdigest()


def write_snapshot(df: pd.DataFrame, name: str) -> tuple[Path, str]:
    """Freeze a training table so a ModelRun can record exactly what it was fit on (data_manifest_hash)."""
    digest = table_hash(df)
    path = Path(settings.data_dir) / "processed" / "features" / name / f"{digest[:16]}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        df.to_parquet(path, index=False)
    return path, digest