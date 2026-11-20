"""Contracts for parsed EONET frames, checked before anything is written."""
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

UTC = {"unit": "ns", "tz": "UTC"}


class EonetEvents(pa.DataFrameModel):
    source_event_id: Series[str] = pa.Field(unique=True, str_length={"min_value": 1, "max_value": 64})
    category: Series[str] = pa.Field(str_length={"min_value": 1, "max_value": 32})
    title: Series[str] = pa.Field(str_length={"min_value": 1})
    is_closed: Series[bool]
    closed_at: Series[pd.DatetimeTZDtype] = pa.Field(nullable=True, dtype_kwargs=UTC)
    first_date: Series[pd.Timestamp]
    last_date: Series[pd.Timestamp]
    lon: Series[float] = pa.Field(ge=-180, le=180)
    lat: Series[float] = pa.Field(ge=-90, le=90)

    class Config:
        coerce = True

    @pa.dataframe_check
    def dates_ordered(cls, df: pd.DataFrame) -> Series[bool]:
        return df["first_date"] <= df["last_date"]

    @pa.dataframe_check
    def closed_flag_matches_date(cls, df: pd.DataFrame) -> Series[bool]:
        return df["is_closed"] == df["closed_at"].notna()


class EonetGeometries(pa.DataFrameModel):
    source_event_id: Series[str]
    observed_at: Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs=UTC)
    geom_type: Series[str] = pa.Field(isin=["Point", "Polygon"])
    wkt: Series[str]
    magnitude_value: Series[float] = pa.Field(nullable=True)
    magnitude_unit: Series[str] = pa.Field(nullable=True, str_length={"max_value": 16})

    class Config:
        coerce = True
