from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db import Base


class Source(Base):
    """A data provider, e.g. 'eonet', 'firms_viirs_snpp', 'gpm_imerg_daily'."""
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(255))
    url: Mapped[str | None] = mapped_column(Text)
    license: Mapped[str | None] = mapped_column(String(255))

    ingest_runs: Mapped[list["IngestRun"]] = relationship(back_populates="source")



class IngestRun(Base):
    """One execution of an ingest pipeline; every ingested row points here."""
    __tablename__ = "ingest_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="running")  # running | ok | failed
    params: Mapped[dict | None] = mapped_column(JSONB)       # region, date range, api args
    rows_in: Mapped[int | None] = mapped_column(Integer)
    rows_written: Mapped[int | None] = mapped_column(Integer)
    raw_path: Mapped[str | None] = mapped_column(Text)        # data/raw/<source>/<date>/...
    error: Mapped[str | None] = mapped_column(Text)
    source: Mapped["Source"] = relationship(back_populates="ingest_runs")