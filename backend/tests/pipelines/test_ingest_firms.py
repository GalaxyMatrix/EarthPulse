from datetime import date

import h3
import pandas as pd
import pytest
from pandera.errors import SchemaError

from pipelines.ingest.firms import ensure_csv, parse, plan

HEADER = "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,instrument,confidence,version,bright_ti5,frp,daynight"
ROW_NIGHT = "38.123456,23.654321,330.5,0.39,0.36,2024-08-11,45,N,VIIRS,n,2.0NRT,290.1,5.2,N"
ROW_DAY = "38.2,23.7,367.0,0.4,0.37,2024-08-11,1130,N,VIIRS,h,2.0NRT,300.2,41.7,D"
CSV = "\n".join([HEADER, ROW_NIGHT, ROW_NIGHT, ROW_DAY])


def test_acq_time_is_zero_padded_to_utc():
    df = parse(CSV)
    assert df["acq_datetime"].tolist() == [
        pd.Timestamp("2024-08-11 00:45", tz="UTC"),
        pd.Timestamp("2024-08-11 11:30", tz="UTC"),
    ]


def test_duplicate_rows_are_dropped():
    assert len(parse(CSV)) == 2


def test_coordinates_rounded_and_h3_filled():
    df = parse(CSV)
    assert df.loc[0, "lat"] == 38.12346
    assert df.loc[0, "lon"] == 23.65432
    assert df.loc[0, "h3_r5"] == h3.latlng_to_cell(38.12346, 23.65432, 5)


def test_header_only_csv_gives_empty_frame():
    df = parse(HEADER)
    assert df.empty
    assert "h3_r5" in df.columns


def test_schema_rejects_impossible_latitude():
    with pytest.raises(SchemaError):
        parse("\n".join([HEADER, ROW_NIGHT.replace("38.123456", "95.0")]))


def test_plain_text_error_body_is_not_treated_as_csv():
    with pytest.raises(RuntimeError, match="Invalid MAP_KEY"):
        ensure_csv("Invalid MAP_KEY.")


def test_plan_uses_archive_then_near_real_time():
    avail = {
        "VIIRS_SNPP_SP": (date(2012, 1, 20), date(2026, 6, 30)),
        "VIIRS_SNPP_NRT": (date(2026, 6, 1), date(2026, 11, 17)),
    }
    assert plan(date(2026, 1, 1), date(2026, 11, 17), ["VIIRS_SNPP_SP", "VIIRS_SNPP_NRT"], avail) == [
        ("VIIRS_SNPP_SP", date(2026, 1, 1), date(2026, 6, 30)),
        ("VIIRS_SNPP_NRT", date(2026, 7, 1), date(2026, 11, 17)),
    ]
