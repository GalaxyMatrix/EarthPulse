from datetime import date, datetime

from geoalchemy2 import Geometry
from sqlalchemy import Date, DateTime, Float, ForeignKey, Index, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class FireDetection(Base):
    """One FIRMS hotspot row (VIIRS or MODIS)."""
    __tablename__ = "fire_detections"
    __table_args__ = (
        # FIRMS has no ID; natural key = satellite + time + location
        UniqueConstraint("satellite", "acq_datetime", "lat", "lon", name="uq_fire_detections_natural"),
        Index("ix_fire_detections_acq_date", "acq_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    ingest_run_id: Mapped[int] = mapped_column(ForeignKey("ingest_runs.id"))
    acq_date: Mapped[date] = mapped_column(Date)
    acq_datetime: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    lat: Mapped[float] = mapped_column(Numeric(8, 5, asdecimal=False))
    lon: Mapped[float] = mapped_column(Numeric(8, 5, asdecimal=False))
    geom: Mapped[object] = mapped_column(Geometry("POINT", srid=4326, spatial_index=True))
    brightness: Mapped[float | None] = mapped_column(Float)   # bright_ti4 (VIIRS) / brightness (MODIS)
    bright_t31: Mapped[float | None] = mapped_column(Float)   # bright_ti5 / bright_t31
    frp: Mapped[float | None] = mapped_column(Float)
    scan: Mapped[float | None] = mapped_column(Float)
    track: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[str | None] = mapped_column(String(8))  # VIIRS: l/n/h ; MODIS: 0-100 stored as text
    daynight: Mapped[str | None] = mapped_column(String(1))
    satellite: Mapped[str] = mapped_column(String(8))          # N, N20, N21, T, A
    instrument: Mapped[str] = mapped_column(String(8))         # VIIRS, MODIS
    version: Mapped[str | None] = mapped_column(String(16))
    h3_r5: Mapped[str | None] = mapped_column(String(16), index=True)  # for baseline features + low-zoom aggregation

    anomaly_scores: Mapped[list["FireAnomalyScore"]] = relationship(back_populates="detection")


class FireCluster(Base):
    """Daily DBSCAN cluster of detections."""
    __tablename__ = "fire_clusters"
    __table_args__ = (
        UniqueConstraint("model_run_id", "cluster_date", "cluster_label", name="uq_fire_clusters_run_date_label"),
        Index("ix_fire_clusters_date", "cluster_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"))
    cluster_date: Mapped[date] = mapped_column(Date)
    cluster_label: Mapped[int] = mapped_column(Integer)       # DBSCAN label within that run/day
    hull: Mapped[object] = mapped_column(Geometry("POLYGON", srid=4326, spatial_index=True))
    centroid: Mapped[object] = mapped_column(Geometry("POINT", srid=4326, spatial_index=True))
    n_detections: Mapped[int] = mapped_column(Integer)
    frp_sum: Mapped[float | None] = mapped_column(Float)
    frp_max: Mapped[float | None] = mapped_column(Float)
    area_km2: Mapped[float | None] = mapped_column(Float)
    matched_event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id"))  # EONET wildfire, if any

    members: Mapped[list["FireClusterMember"]] = relationship(back_populates="cluster", cascade="all, delete-orphan")


class FireClusterMember(Base):
    __tablename__ = "fire_cluster_members"

    cluster_id: Mapped[int] = mapped_column(ForeignKey("fire_clusters.id", ondelete="CASCADE"), primary_key=True)
    detection_id: Mapped[int] = mapped_column(ForeignKey("fire_detections.id"), primary_key=True)

    cluster: Mapped["FireCluster"] = relationship(back_populates="members")


class FireAnomalyScore(Base):
    """Isolation Forest score for a detection under a given model run."""
    __tablename__ = "fire_anomaly_scores"
    __table_args__ = (
        UniqueConstraint("detection_id", "model_run_id", name="uq_fire_anomaly_detection_run"),
        Index("ix_fire_anomaly_run_rank", "model_run_id", "rank_in_day"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    detection_id: Mapped[int] = mapped_column(ForeignKey("fire_detections.id"))
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"))
    score: Mapped[float] = mapped_column(Float)                # higher = more anomalous (negate sklearn output)
    percentile: Mapped[float | None] = mapped_column(Float)    # within that day's scored population
    rank_in_day: Mapped[int | None] = mapped_column(Integer)
    feature_contrib: Mapped[dict | None] = mapped_column(JSONB)  # {"frp": {"value": 312, "baseline_p50": 18, "z": 4.1}, ...}

    detection: Mapped["FireDetection"] = relationship(back_populates="anomaly_scores")
