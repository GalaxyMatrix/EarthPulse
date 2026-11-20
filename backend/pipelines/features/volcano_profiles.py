from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Volcano, VolcanoEruption
from ml.volcano.timeline import TimelineParams, repose_percentile, volcano_profile

PROFILE_COLUMNS = [
    "n_events", "first_event", "last_event", "is_ongoing", "is_active", "days_since_last",
    "median_repose_days", "median_duration_days", "events_per_decade", "max_vei", "repose_percentile",
]


def load_volcanoes(session: Session) -> pd.DataFrame:
    stmt = select(
        Volcano.id.label("volcano_id"),
        Volcano.name,
        Volcano.volcano_type,
        Volcano.tectonic_setting,
        Volcano.elevation_m,
        Volcano.lat,
        Volcano.lon,
    )
    return pd.read_sql(stmt, session.connection())


def load_eruptions(session: Session, reference_date: date) -> pd.DataFrame:
    """Eruptions as they were known on reference_date: later starts are dropped, later ends count as ongoing."""
    stmt = select(
        VolcanoEruption.volcano_id,
        VolcanoEruption.start_date,
        VolcanoEruption.end_date,
        VolcanoEruption.vei,
    ).where(VolcanoEruption.start_date <= reference_date)
    e = pd.read_sql(stmt, session.connection())
    e.loc[pd.to_datetime(e["end_date"]) > pd.Timestamp(reference_date), "end_date"] = None
    return e


def build_profiles(
    volcanoes: pd.DataFrame, eruptions: pd.DataFrame, reference_date: date, params: TimelineParams = TimelineParams()
) -> pd.DataFrame:
    """One row per volcano with at least one eruption: metadata + timeline profile at reference_date."""
    rows = [
        {
            "volcano_id": int(vid),
            **volcano_profile(ev, reference_date, params),
            "repose_percentile": repose_percentile(ev, reference_date),
        }
        for vid, ev in eruptions.groupby("volcano_id")
    ]
    profiles = pd.DataFrame(rows, columns=["volcano_id", *PROFILE_COLUMNS])
    profiles["reference_date"] = reference_date
    return volcanoes.merge(profiles, on="volcano_id", how="inner")


def load_profile_table(
    session: Session, reference_date: date, params: TimelineParams = TimelineParams()
) -> pd.DataFrame:
    """Input for ml.volcano.clustering.fit_assign and for writing VolcanoProfile rows."""
    return build_profiles(load_volcanoes(session), load_eruptions(session, reference_date), reference_date, params)