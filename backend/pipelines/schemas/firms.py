"""Contract for normalized FIRMS VIIRS rows (after column mapping, before h3 and the upsert)."""
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

NATURAL_KEY = ["satellite", "acq_datetime", "lat", "lon"]


class FirmsDetections(pa.DataFrameModel):
    acq_datetime: Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"unit": "ns", "tz": "UTC"})
    lat: Series[float] = pa.Field(ge=-90, le=90)
    lon: Series[float] = pa.Field(ge=-180, le=180)
    brightness: Series[float] = pa.Field(gt=150, lt=500, nullable=True)
    bright_t31: Series[float] = pa.Field(gt=150, lt=500, nullable=True)
    frp: Series[float] = pa.Field(ge=0, nullable=True)
    scan: Series[float] = pa.Field(gt=0, nullable=True)
    track: Series[float] = pa.Field(gt=0, nullable=True)
    confidence: Series[str] = pa.Field(isin=["l", "n", "h"], nullable=True)
    daynight: Series[str] = pa.Field(isin=["D", "N"])
    satellite: Series[str] = pa.Field(isin=["N", "N20", "N21"])
    instrument: Series[str] = pa.Field(isin=["VIIRS"])
    version: Series[str] = pa.Field(nullable=True, str_length={"max_value": 16})

    class Config:
        coerce = True

    @pa.dataframe_check
    def natural_key_unique(cls, df: pd.DataFrame) -> bool:
        return not df.duplicated(NATURAL_KEY).any()
