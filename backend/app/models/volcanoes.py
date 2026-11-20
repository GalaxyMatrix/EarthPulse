from datetime import date

from geoalchemy2 import Geometry
from sqlalchemy import Boolean, Date, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Volcano(Base):
    """A volcano from the Smithsonian GVP catalogue."""
    __tablename__ = "volcanoes"
    __table_args__ = (UniqueConstraint("source_id", "source_volcano_id", name="uq_volcanoes_source_volcano"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    ingest_run_id: Mapped[int] = mapped_column(ForeignKey("ingest_runs.id"))
    source_volcano_id: Mapped[str] = mapped_column(String(32))       # GVP volcano number
    name: Mapped[str] = mapped_column(String(128), index=True)
    country: Mapped[str | None] = mapped_column(String(64))
    volcano_type: Mapped[str | None] = mapped_column(String(64))     # stratovolcano, shield, caldera, ...
    tectonic_setting: Mapped[str | None] = mapped_column(String(64))
    elevation_m: Mapped[float | None] = mapped_column(Float)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    geom: Mapped[object] = mapped_column(Geometry("POINT", srid=4326, spatial_index=True))

    eruptions: Mapped[list["VolcanoEruption"]] = relationship(
        back_populates="volcano", cascade="all, delete-orphan", order_by="VolcanoEruption.start_date"
    )


class VolcanoEruption(Base):
    """One eruption episode (GVP), optionally linked to the matching EONET event."""
    __tablename__ = "volcano_eruptions"
    __table_args__ = (
        UniqueConstraint("source_id", "source_eruption_id", name="uq_volcano_eruptions_source_eruption"),
        Index("ix_volcano_eruptions_volcano_start", "volcano_id", "start_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    volcano_id: Mapped[int] = mapped_column(ForeignKey("volcanoes.id", ondelete="CASCADE"))
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    ingest_run_id: Mapped[int] = mapped_column(ForeignKey("ingest_runs.id"))
    source_eruption_id: Mapped[str] = mapped_column(String(32))
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)              # null = ongoing
    vei: Mapped[float | None] = mapped_column(Float)
    evidence: Mapped[str | None] = mapped_column(String(64))
    event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id"))  # matching EONET volcano event, if any

    volcano: Mapped["Volcano"] = relationship(back_populates="eruptions")


class VolcanoProfile(Base):
    """Timeline statistics for one volcano at a reference date (ml.volcano.timeline). Descriptive only."""
    __tablename__ = "volcano_profiles"
    __table_args__ = (
        UniqueConstraint("volcano_id", "model_run_id", "reference_date", name="uq_volcano_profile_volcano_run_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    volcano_id: Mapped[int] = mapped_column(ForeignKey("volcanoes.id", ondelete="CASCADE"), index=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"))
    reference_date: Mapped[date] = mapped_column(Date)
    n_events: Mapped[int] = mapped_column(Integer)
    first_event: Mapped[date | None] = mapped_column(Date)
    last_event: Mapped[date | None] = mapped_column(Date)
    is_ongoing: Mapped[bool] = mapped_column(Boolean)
    is_active: Mapped[bool] = mapped_column(Boolean)
    days_since_last: Mapped[int | None] = mapped_column(Integer)
    median_repose_days: Mapped[float | None] = mapped_column(Float)
    median_duration_days: Mapped[float | None] = mapped_column(Float)
    events_per_decade: Mapped[float | None] = mapped_column(Float)
    max_vei: Mapped[float | None] = mapped_column(Float)
    repose_percentile: Mapped[float | None] = mapped_column(Float)   # not an eruption probability


class VolcanoCluster(Base):
    """An agglomerative cluster of volcanoes by activity profile + metadata under one run."""
    __tablename__ = "volcano_clusters"
    __table_args__ = (UniqueConstraint("model_run_id", "cluster_label", name="uq_volcano_cluster_run_label"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"), index=True)
    cluster_label: Mapped[int] = mapped_column(Integer)
    name: Mapped[str | None] = mapped_column(String(64))
    n_volcanoes: Mapped[int] = mapped_column(Integer)
    top_type: Mapped[str | None] = mapped_column(String(64))
    top_setting: Mapped[str | None] = mapped_column(String(64))
    profile: Mapped[dict | None] = mapped_column(JSONB)              # median numeric features (describe_clusters)
    description: Mapped[str | None] = mapped_column(Text)

    members: Mapped[list["VolcanoClusterMember"]] = relationship(back_populates="cluster", cascade="all, delete-orphan")


class VolcanoClusterMember(Base):
    __tablename__ = "volcano_cluster_members"

    cluster_id: Mapped[int] = mapped_column(ForeignKey("volcano_clusters.id", ondelete="CASCADE"), primary_key=True)
    volcano_id: Mapped[int] = mapped_column(ForeignKey("volcanoes.id", ondelete="CASCADE"), primary_key=True)

    cluster: Mapped["VolcanoCluster"] = relationship(back_populates="members")
