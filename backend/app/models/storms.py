from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

# Storm events and their track points live in `events` / `event_geometries` (EONET severeStorms, IBTrACS).
# These tables hold what ml/storms derives from them.


class StormTrackMetric(Base):
    """Per track point metrics for one storm (ml.storms.tracks.track_metrics)."""
    __tablename__ = "storm_track_metrics"
    __table_args__ = (
        UniqueConstraint("event_geometry_id", "model_run_id", name="uq_storm_track_metric_point_run"),
        Index("ix_storm_track_metrics_event_time", "event_id", "observed_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"))
    event_geometry_id: Mapped[int] = mapped_column(ForeignKey("event_geometries.id", ondelete="CASCADE"))
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"))
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    dt_h: Mapped[float | None] = mapped_column(Float)
    dist_km: Mapped[float | None] = mapped_column(Float)
    speed_kmh: Mapped[float | None] = mapped_column(Float)
    heading_deg: Mapped[float | None] = mapped_column(Float)
    d_intensity_kts: Mapped[float | None] = mapped_column(Float)
    intensity_change_window: Mapped[float | None] = mapped_column(Float)  # change over the RI window (24 h by default)
    rapid_intensification: Mapped[bool] = mapped_column(Boolean, default=False)


class StormSummary(Base):
    """One row per storm per run: summary stats + shape features (storm_summary, track_shape_features)."""
    __tablename__ = "storm_summaries"
    __table_args__ = (UniqueConstraint("event_id", "model_run_id", name="uq_storm_summary_event_run"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), index=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"))
    n_points: Mapped[int] = mapped_column(Integer)
    duration_h: Mapped[float] = mapped_column(Float)
    track_length_km: Mapped[float] = mapped_column(Float)
    net_displacement_km: Mapped[float | None] = mapped_column(Float)
    sinuosity: Mapped[float | None] = mapped_column(Float)
    mean_heading_deg: Mapped[float | None] = mapped_column(Float)
    mean_speed_kmh: Mapped[float | None] = mapped_column(Float)
    peak_intensity_kts: Mapped[float | None] = mapped_column(Float)
    peak_intensity_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    n_ri_points: Mapped[int] = mapped_column(Integer, default=0)
    max_window_intensification_kts: Mapped[float | None] = mapped_column(Float)
    magnitude_coverage: Mapped[float] = mapped_column(Float)  # share of track points with an intensity value


class StormCluster(Base):
    """A k-means cluster of storms by track shape / intensity under one run."""
    __tablename__ = "storm_clusters"
    __table_args__ = (UniqueConstraint("model_run_id", "cluster_label", name="uq_storm_cluster_run_label"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"), index=True)
    cluster_label: Mapped[int] = mapped_column(Integer)
    name: Mapped[str | None] = mapped_column(String(64))       # human label for the legend, e.g. "long-track intense"
    n_storms: Mapped[int] = mapped_column(Integer)
    profile: Mapped[dict | None] = mapped_column(JSONB)         # median feature values (describe_clusters)

    members: Mapped[list["StormClusterMember"]] = relationship(back_populates="cluster", cascade="all, delete-orphan")


class StormClusterMember(Base):
    __tablename__ = "storm_cluster_members"

    cluster_id: Mapped[int] = mapped_column(ForeignKey("storm_clusters.id", ondelete="CASCADE"), primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"), primary_key=True)
    distance_to_center: Mapped[float | None] = mapped_column(Float)

    cluster: Mapped["StormCluster"] = relationship(back_populates="members")
