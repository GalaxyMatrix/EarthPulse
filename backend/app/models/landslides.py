from datetime import date

from geoalchemy2 import Geometry
from sqlalchemy import Date, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class LandslideEvent(Base):
    """A reported landslide from NASA's Global Landslide Catalog / COOLR, optionally linked to EONET."""
    __tablename__ = "landslide_events"
    __table_args__ = (
        UniqueConstraint("source_id", "source_event_id", name="uq_landslide_events_source_event"),
        Index("ix_landslide_events_date", "event_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    ingest_run_id: Mapped[int] = mapped_column(ForeignKey("ingest_runs.id"))
    source_event_id: Mapped[str] = mapped_column(String(64))
    event_date: Mapped[date] = mapped_column(Date)
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    geom: Mapped[object] = mapped_column(Geometry("POINT", srid=4326, spatial_index=True))
    location_accuracy: Mapped[str | None] = mapped_column(String(32))  # GLC: exact, 1km, 5km, 25km, ...
    country: Mapped[str | None] = mapped_column(String(64))
    title: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(32))           # landslide, mudslide, debris_flow, ...
    trigger: Mapped[str | None] = mapped_column(String(64))            # raw GLC trigger label
    size: Mapped[str | None] = mapped_column(String(32))
    setting: Mapped[str | None] = mapped_column(String(64))
    fatalities: Mapped[int | None] = mapped_column(Integer)
    source_link: Mapped[str | None] = mapped_column(Text)
    event_id: Mapped[int | None] = mapped_column(ForeignKey("events.id"))  # matching EONET event, if any

    features: Mapped[list["LandslideFeature"]] = relationship(back_populates="event", cascade="all, delete-orphan")
    predictions: Mapped[list["LandslidePrediction"]] = relationship(back_populates="event", cascade="all, delete-orphan")


class LandslideFeature(Base):
    """Antecedent rainfall, terrain and land-cover features for one event (ml.landslide.features)."""
    __tablename__ = "landslide_features"
    __table_args__ = (UniqueConstraint("landslide_event_id", "feature_version", name="uq_landslide_features_event_version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    landslide_event_id: Mapped[int] = mapped_column(ForeignKey("landslide_events.id", ondelete="CASCADE"), index=True)
    feature_version: Mapped[str] = mapped_column(String(32))          # bump when feature code or inputs change
    rain_1d_mm: Mapped[float | None] = mapped_column(Float)
    rain_3d_mm: Mapped[float | None] = mapped_column(Float)
    rain_7d_mm: Mapped[float | None] = mapped_column(Float)
    rain_14d_mm: Mapped[float | None] = mapped_column(Float)
    rain_30d_mm: Mapped[float | None] = mapped_column(Float)
    rain_max_daily_30d: Mapped[float | None] = mapped_column(Float)
    api: Mapped[float | None] = mapped_column(Float)                  # antecedent precipitation index
    slope_deg: Mapped[float | None] = mapped_column(Float)
    elevation_m: Mapped[float | None] = mapped_column(Float)
    relief_m: Mapped[float | None] = mapped_column(Float)
    land_cover: Mapped[str | None] = mapped_column(String(32))
    extra: Mapped[dict | None] = mapped_column(JSONB)                 # DEM/land-cover product versions, etc.

    event: Mapped["LandslideEvent"] = relationship(back_populates="features")


class LandslideCluster(Base):
    """A spatio-temporal DBSCAN cluster of reported landslides (e.g. a storm-triggered swarm)."""
    __tablename__ = "landslide_clusters"
    __table_args__ = (UniqueConstraint("model_run_id", "cluster_label", name="uq_landslide_cluster_run_label"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"), index=True)
    cluster_label: Mapped[int] = mapped_column(Integer)
    n_events: Mapped[int] = mapped_column(Integer)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    centroid: Mapped[object] = mapped_column(Geometry("POINT", srid=4326, spatial_index=True))
    top_trigger: Mapped[str | None] = mapped_column(String(64))

    members: Mapped[list["LandslideClusterMember"]] = relationship(back_populates="cluster", cascade="all, delete-orphan")


class LandslideClusterMember(Base):
    __tablename__ = "landslide_cluster_members"

    cluster_id: Mapped[int] = mapped_column(ForeignKey("landslide_clusters.id", ondelete="CASCADE"), primary_key=True)
    landslide_event_id: Mapped[int] = mapped_column(ForeignKey("landslide_events.id", ondelete="CASCADE"), primary_key=True)

    cluster: Mapped["LandslideCluster"] = relationship(back_populates="members")


class LandslidePrediction(Base):
    """Predicted trigger class for an event under a classifier run (ml.landslide.classifier.predict)."""
    __tablename__ = "landslide_predictions"
    __table_args__ = (UniqueConstraint("landslide_event_id", "model_run_id", name="uq_landslide_prediction_event_run"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    landslide_event_id: Mapped[int] = mapped_column(ForeignKey("landslide_events.id", ondelete="CASCADE"), index=True)
    model_run_id: Mapped[int] = mapped_column(ForeignKey("model_runs.id", ondelete="CASCADE"))
    predicted_class: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    probabilities: Mapped[dict] = mapped_column(JSONB)                # {"rain": 0.81, "earthquake": 0.12, ...}

    event: Mapped["LandslideEvent"] = relationship(back_populates="predictions")
