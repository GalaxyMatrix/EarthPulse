from datetime import date

from geoalchemy2 import Geometry
from sqlalchemy import Date, Float, ForeignKey, Index, Integer, SmallInteger, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class RainCell(Base):
    """A 0.1° IMERG grid cell that we track as rows (regional subset, not the globe)."""
    __tablename__ = "rain_cells"
    __table_args__ = (UniqueConstraint("row", "col", name="uq_rain_cells_rowcol"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    row: Mapped[int] = mapped_column(Integer)                  # IMERG grid indices
    col: Mapped[int] = mapped_column(Integer)
    region: Mapped[str] = mapped_column(String(32), index=True)  # key from config/regions.yaml
    geom: Mapped[object] = mapped_column(Geometry("POLYGON", srid=4326, spatial_index=True))
    centroid: Mapped[object] = mapped_column(Geometry("POINT", srid=4326, spatial_index=True))

    daily: Mapped[list["RainDaily"]] = relationship(back_populates="cell")


class RainDaily(Base):
    __tablename__ = "rain_daily"
    __table_args__ = (UniqueConstraint("cell_id", "obs_date", name="uq_rain_daily_cell_date"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    cell_id: Mapped[int] = mapped_column(ForeignKey("rain_cells.id"), index=True)
    ingest_run_id: Mapped[int] = mapped_column(ForeignKey("ingest_runs.id"))
    obs_date: Mapped[date] = mapped_column(Date, index=True)
    precip_mm: Mapped[float] = mapped_column(Float)

    cell: Mapped["RainCell"] = relationship(back_populates="daily")


class RainClimatology(Base):
    """Per-cell, per-day-of-year-window baseline statistics."""
    __tablename__ = "rain_climatology"
    __table_args__ = (UniqueConstraint("cell_id", "doy_center", "window_days", name="uq_rain_clim_cell_doy"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    cell_id: Mapped[int] = mapped_column(ForeignKey("rain_cells.id"), index=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"))
    doy_center: Mapped[int] = mapped_column(SmallInteger)      # 1..366
    window_days: Mapped[int] = mapped_column(SmallInteger)     # e.g. 31 (±15)
    n_obs: Mapped[int] = mapped_column(Integer)
    mean_mm: Mapped[float] = mapped_column(Float)
    std_mm: Mapped[float] = mapped_column(Float)
    p50_mm: Mapped[float] = mapped_column(Float)
    p90_mm: Mapped[float] = mapped_column(Float)
    p99_mm: Mapped[float] = mapped_column(Float)


class RainAnomaly(Base):
    __tablename__ = "rain_anomalies"
    __table_args__ = (
        UniqueConstraint("cell_id", "obs_date", "model_run_id", name="uq_rain_anom_cell_date_run"),
        Index("ix_rain_anomalies_date_score", "obs_date", "score"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    cell_id: Mapped[int] = mapped_column(ForeignKey("rain_cells.id"), index=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"))
    obs_date: Mapped[date] = mapped_column(Date)
    precip_mm: Mapped[float] = mapped_column(Float)
    z: Mapped[float | None] = mapped_column(Float)
    percentile: Mapped[float | None] = mapped_column(Float)
    score: Mapped[float] = mapped_column(Float)                # detector-specific; see model card
    window_days: Mapped[int] = mapped_column(SmallInteger, default=1)  # 1, 3, 7-day total
    matched_event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id"))  # EONET flood, if any


class RainForecast(Base):
    """Forecast experiment results, including baselines, so skill can be computed."""
    __tablename__ = "rain_forecasts"
    __table_args__ = (
        UniqueConstraint("cell_id", "issue_date", "target_date", "model_run_id", name="uq_rain_fc"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    cell_id: Mapped[int] = mapped_column(ForeignKey("rain_cells.id"), index=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"))  # one run per model (persistence, climatology, lgbm, ...)
    issue_date: Mapped[date] = mapped_column(Date)
    target_date: Mapped[date] = mapped_column(Date, index=True)
    pred_mm: Mapped[float] = mapped_column(Float)
    actual_mm: Mapped[float | None] = mapped_column(Float)     # filled once observed
