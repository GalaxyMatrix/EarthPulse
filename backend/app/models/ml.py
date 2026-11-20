from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Index, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class ModelRun(Base):
    """One trained/fitted model artefact. Everything ML-derived references a run."""
    __tablename__ = "model_runs"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_model_runs_name_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(64), index=True)   # fire_anomaly_if, fire_clusters_dbscan, rain_climatology, rain_anomaly_z, rain_forecast_lgbm, ...
    version: Mapped[str] = mapped_column(String(32))            # semver or run timestamp
    hazard: Mapped[str] = mapped_column(String(16))             # wildfire | rainfall | storm | volcano | landslide
    region: Mapped[str | None] = mapped_column(String(32))
    params: Mapped[dict | None] = mapped_column(JSONB)          # hyperparameters, feature list
    train_start: Mapped[date | None] = mapped_column(Date)
    train_end: Mapped[date | None] = mapped_column(Date)
    artifact_path: Mapped[str | None] = mapped_column(Text)     # data/processed/models/<name>/<version>.joblib
    model_card_path: Mapped[str | None] = mapped_column(Text)   # docs/model_cards/<name>.md
    notes: Mapped[str | None] = mapped_column(Text)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64), index=True)
    registry_alias: Mapped[str | None] = mapped_column(String(16), index=True)  # champion | challenger | archived
    data_manifest_hash: Mapped[str | None] = mapped_column(String(64))         # hash of the training data snapshot
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    evaluations: Mapped[list["Evaluation"]] = relationship(back_populates="model_run", cascade="all, delete-orphan")
    monitoring: Mapped[list["Monitoring"]] = relationship(back_populates="model_run", cascade="all, delete-orphan")


class Evaluation(Base):
    """A single metric value for a model run, e.g. precision@50 vs EONET wildfires."""
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"), index=True)
    metric: Mapped[str] = mapped_column(String(64))             # precision_at_k, hit_rate, mae, skill_vs_climatology, ...
    value: Mapped[float] = mapped_column(Float)
    eval_start: Mapped[date | None] = mapped_column(Date)
    eval_end: Mapped[date | None] = mapped_column(Date)
    baseline: Mapped[str | None] = mapped_column(String(64))    # what this is compared against, if anything
    details: Mapped[dict | None] = mapped_column(JSONB)         # k, buffer_km, time_window_days, n_matched, ...
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    model_run: Mapped["ModelRun"] = relationship(back_populates="evaluations")


class Monitoring(Base):
    """A post-deployment check on one scoring run: input drift, score-distribution shift, catalogue match rate."""
    __tablename__ = "monitoring"
    __table_args__ = (Index("ix_monitoring_run_date_metric", "model_run_id", "scored_date", "metric"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"))
    scored_date: Mapped[date] = mapped_column(Date)
    metric: Mapped[str] = mapped_column(String(64))             # feature_drift_share, score_psi, eonet_match_rate_7d, ...
    value: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)
    breached: Mapped[bool] = mapped_column(Boolean, default=False)  # read by flows/retrain_check.py
    report_path: Mapped[str | None] = mapped_column(Text)       # Evidently HTML/JSON artefact
    details: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    model_run: Mapped["ModelRun"] = relationship(back_populates="monitoring")
