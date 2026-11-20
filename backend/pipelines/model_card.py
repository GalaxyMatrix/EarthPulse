"""Load a ModelRun + evaluations and write docs/model_cards/<name>.md."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import session_scope
from app.models import Evaluation, ModelRun
from ml.common.model_card import render

CARDS = {
    "fire_anomaly_if": {
        "method": "Isolation Forest on FIRMS VIIRS features + H3×month FRP baseline",
        "intended_use": (
            "Rank wildfire hotspot detections by how unusual they are relative to the same "
            "H3 cell and calendar month, to surface candidates for analyst review."
        ),
        "not_for": (
            "Operational fire alerts, ignition prediction, burned-area mapping, or comparison "
            "of scores with other hazards."
        ),
        "data": {
            "detections": "NASA FIRMS VIIRS S-NPP 375 m",
            "reference": "NASA EONET wildfires (evaluation only)",
            "features": "frp, brightness, bright_t31, scan, track, cell FRP z/ratio, day/night, confidence",
        },
        "limitations": [
            "Scores are relative within the scored window, not calibrated probabilities.",
            "feature_contrib is a standardized deviation view, not SHAP attribution.",
            "New H3 cells without baseline history get neutral cell features.",
            "Evaluation against EONET is a coarse spatial-temporal match, not ground truth.",
        ],
    },
    "fire_clusters_dbscan": {
        "method": "Daily DBSCAN (haversine) on FIRMS detection coordinates",
        "intended_use": (
            "Group same-day hotspots into spatial clusters for map display and comparison "
            "with catalogue wildfire events."
        ),
        "not_for": "Perimeter mapping, fire-spread modelling, or counting distinct ignitions.",
        "data": {
            "detections": "NASA FIRMS VIIRS S-NPP 375 m",
            "reference": "NASA EONET wildfires (evaluation only)",
        },
        "limitations": [
            "eps_km and min_samples are heuristic; noise points are dropped.",
            "Hulls are convex buffers, not burned area.",
            "Clustering is day-scoped; multi-day fires appear as separate daily clusters.",
        ],
    },
    "rain_climatology": {
        "method": "Per-cell day-of-year window mean/std and quantiles",
        "intended_use": "Provide a seasonal baseline for rainfall anomaly and forecast features.",
        "not_for": "Direct flood warnings or climate-change trend analysis.",
        "data": {"precip": "GPM IMERG Final daily (regional cells)"},
        "limitations": [
            "Needs multiple years per cell for stable quantiles (min_years).",
            "Window wraps at year-end; leap days are included in DOY 366.",
        ],
    },
    "rain_anomaly_z": {
        "method": "Rolling precip totals vs DOY climatology (z-score / percentile)",
        "intended_use": (
            "Flag unusually wet cell-days relative to the local seasonal baseline for flood "
            "candidate review."
        ),
        "not_for": "Flash-flood warnings, urban drainage design, or drought monitoring.",
        "data": {
            "precip": "GPM IMERG Final daily",
            "baseline": "rain_climatology ModelRun",
            "reference": "NASA EONET floods (evaluation only)",
        },
        "limitations": [
            "Only one window_days is stored per run (DB unique key).",
            "IMERG Final lags real time by months.",
            "EONET floods are sparse and spatially coarse versus 0.1° cells.",
        ],
    },
    "rain_forecast_lgbm": {
        "method": "LightGBM (tweedie) next-day precip with persistence/climatology/roll7 baselines",
        "intended_use": "Research comparison of short-range precip skill on regional IMERG cells.",
        "not_for": "Operational weather forecasting or hydrological routing.",
        "data": {"precip": "GPM IMERG Final daily", "baseline": "rain_climatology features"},
        "limitations": [
            "Strict time split; no leakage across train_end.",
            "Skill is region- and period-specific.",
        ],
    },
}


def _meta(name: str) -> dict:
    if name not in CARDS:
        return {
            "method": name,
            "intended_use": "See repository docs.",
            "not_for": "Safety-critical decisions.",
            "data": {},
            "limitations": ["Card template was not specialized for this model name."],
        }
    return CARDS[name]


def build_markdown(session: Session, run: ModelRun) -> str:
    meta = _meta(run.name)
    evals = session.execute(
        select(Evaluation)
        .where(Evaluation.model_run_id == run.id)
        .order_by(Evaluation.metric)
    ).scalars().all()
    return render(
        name=run.name,
        version=run.version,
        hazard=run.hazard,
        method=meta["method"],
        intended_use=meta["intended_use"],
        not_for=meta["not_for"],
        data={
            **meta["data"],
            "region": run.region or "—",
            "params_keys": ", ".join((run.params or {}).keys()) or "—",
            "artifact": run.artifact_path or "—",
            "data_manifest_hash": (run.data_manifest_hash or "—")[:16],
        },
        params=run.params or {},
        evaluations=[
            {
                "metric": e.metric,
                "value": e.value,
                "baseline": e.baseline,
                "details": e.details or {},
            }
            for e in evals
        ],
        limitations=meta["limitations"],
        region=run.region,
        train_start=run.train_start,
        train_end=run.train_end,
        mlflow_run_id=run.mlflow_run_id,
    )


def write_card(run_id: int) -> Path:
    """Render the card, write it under docs/model_cards/, and set model_card_path."""
    with session_scope() as session:
        run = session.get(ModelRun, run_id)
        if run is None:
            raise ValueError(f"No model run {run_id}")
        md = build_markdown(session, run)
        # settings.data_dir is ../data from backend/; cards live in repo docs/
        root = Path(settings.config_dir).resolve().parent  # .../EarthPulse
        path = root / "docs" / "model_cards" / f"{run.name}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
        run.model_card_path = str(path.relative_to(root)).replace("\\", "/")
        return path
