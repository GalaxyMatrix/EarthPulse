from datetime import date, datetime

from geoalchemy2 import Geometry
from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


class Event(Base):
    """Catalogue event from EONET (later GVP, COOLR). One row per event."""
    __tablename__ = "events"
    __table_args__ = (
        UniqueConstraint("source_id", "source_event_id", name="uq_events_source_event"),
        Index("ix_events_category_dates", "category", "first_date", "last_date"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    ingest_run_id: Mapped[int] = mapped_column(ForeignKey("ingest_runs.id"))
    source_event_id: Mapped[str] = mapped_column(String(64))      # EONET "EONET_1234"
    category: Mapped[str] = mapped_column(String(32), index=True) # wildfires, severeStorms, volcanoes, floods, landslides, ...
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    link: Mapped[str | None] = mapped_column(Text)
    is_closed: Mapped[bool] = mapped_column(Boolean, default=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    first_date: Mapped[date] = mapped_column(Date)
    last_date: Mapped[date] = mapped_column(Date)
    # Representative geometry for map rendering at low zoom (first point or centroid)
    geom: Mapped[object] = mapped_column(Geometry("POINT", srid=4326, spatial_index=True))
    extra: Mapped[dict | None] = mapped_column(JSONB)              # EONET sources[], categories[], etc.

    geometries: Mapped[list["EventGeometry"]] = relationship(
        back_populates="event", cascade="all, delete-orphan", order_by="EventGeometry.observed_at"
    )


class EventGeometry(Base):
    """One EONET geometry entry: a point/polygon at a time, with optional magnitude (cyclone tracks)."""
    __tablename__ = "event_geometries"
    __table_args__ = (
        Index("ix_event_geometries_event_time", "event_id", "observed_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id", ondelete="CASCADE"))
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    geom: Mapped[object] = mapped_column(Geometry("GEOMETRY", srid=4326, spatial_index=True))  # POINT or POLYGON
    magnitude_value: Mapped[float | None] = mapped_column(Float)
    magnitude_unit: Mapped[str | None] = mapped_column(String(16))  # "kts", "NM", ...

    event: Mapped["Event"] = relationship(back_populates="geometries")
