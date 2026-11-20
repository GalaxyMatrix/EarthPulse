from __future__ import annotations
import re
from dataclasses import dataclass, fields
from datetime import date, timedelta
import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from app.models import Event, EventGeometry, ModelRun, Source
from ml.storms.tracks import StormParams, storm_summary
from pipelines.evaluate import with_context, write_evaluations



CATEGORY = "severeStorms"
GENERIC_WORDS = {
    "HURRICANE", "TYPHOON", "TROPICAL", "SUBTROPICAL", "STORM", "CYCLONE", "DEPRESSION",
    "SUPER", "SEVERE", "POST", "REMNANTS", "OF", "INVEST",
}


@dataclass(frozen=True)
class StormEvalParams:
    max_start_gap_days: int = 3



