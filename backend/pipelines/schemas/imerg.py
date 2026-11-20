"""Contract for one or more days of IMERG cell values, keyed by grid index (before cell_id lookup)."""
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


class ImergDaily(pa.DataFrameModel):
    row: Series[int] = pa.Field(ge=0, lt=1800)
    col: Series[int] = pa.Field(ge=0, lt=3600)
    obs_date: Series[pd.Timestamp]
    precip_mm: Series[float] = pa.Field(ge=0, le=2000)

    class Config:
        coerce = True

    @pa.dataframe_check
    def one_value_per_cell_day(cls, df: pd.DataFrame) -> bool:
        return not df.duplicated(["row", "col", "obs_date"]).any()
