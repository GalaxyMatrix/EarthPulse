import pandas as pd
import pytest
from pandera.errors import SchemaError

from pipelines.ingest.eonet import merge_events, parse
from pipelines.schemas.eonet import EonetEvents

SQUARE = [[[23.0, 38.0], [24.0, 38.0], [24.0, 39.0], [23.0, 39.0], [23.0, 38.0]]]


def _event(id_, category, geometry, closed=None):
    return {
        "id": id_, "title": f"Event {id_}", "description": None,
        "link": f"https://eonet.gsfc.nasa.gov/api/v3/events/{id_}", "closed": closed,
        "categories": [{"id": category, "title": category}],
        "sources": [{"id": "TEST", "url": "https://example.org"}],
        "geometry": geometry,
    }


def _point(when, lon, lat, mag=None, unit=None):
    return {"date": when, "type": "Point", "coordinates": [lon, lat], "magnitudeValue": mag, "magnitudeUnit": unit}


EVENTS = [
    _event("EONET_1", "wildfires", [_point("2024-08-11T14:00:00Z", 23.9, 38.1)], closed="2024-08-20T00:00:00Z"),
    _event("EONET_2", "severeStorms", [
        _point("2024-09-02T06:00:00Z", 21.0, 36.5, 50.0, "kts"),
        _point("2024-09-01T18:00:00Z", 20.0, 36.0, 35.0, "kts"),
    ]),
    _event("EONET_3", "floods", [{"date": "2024-09-05T00:00:00Z", "type": "Polygon", "coordinates": SQUARE}]),
    _event("EONET_4", "volcanoes", []),
]


@pytest.fixture
def parsed():
    events, geoms = parse(EVENTS)
    return events.set_index("source_event_id"), geoms


def test_one_row_per_event_with_geometry(parsed):
    events, _ = parsed
    assert sorted(events.index) == ["EONET_1", "EONET_2", "EONET_3"]
    assert events.loc["EONET_2", "category"] == "severeStorms"


def test_dates_span_all_geometries(parsed):
    events, _ = parsed
    assert events.loc["EONET_2", "first_date"] == pd.Timestamp("2024-09-01")
    assert events.loc["EONET_2", "last_date"] == pd.Timestamp("2024-09-02")


def test_closed_events(parsed):
    events, _ = parsed
    assert events.loc["EONET_1", "is_closed"]
    assert events.loc["EONET_1", "closed_at"] == pd.Timestamp("2024-08-20", tz="UTC")
    assert not events.loc["EONET_2", "is_closed"]
    assert pd.isna(events.loc["EONET_2", "closed_at"])


def test_polygon_event_is_placed_at_its_centroid(parsed):
    events, geoms = parsed
    assert (events.loc["EONET_3", "lon"], events.loc["EONET_3", "lat"]) == pytest.approx((23.5, 38.5))
    assert geoms.loc[geoms["source_event_id"] == "EONET_3", "geom_type"].tolist() == ["Polygon"]


def test_storm_track_is_time_ordered_with_magnitudes(parsed):
    _, geoms = parsed
    track = geoms[geoms["source_event_id"] == "EONET_2"]
    assert track["observed_at"].is_monotonic_increasing
    assert track["magnitude_value"].tolist() == [35.0, 50.0]
    assert set(track["magnitude_unit"]) == {"kts"}


def test_merge_keeps_union_of_geometries_across_windows():
    early = _event("EONET_9", "wildfires", [_point("2024-01-30T00:00:00Z", 23.0, 38.0)])
    late = _event("EONET_9", "wildfires", [
        _point("2024-01-30T00:00:00Z", 23.0, 38.0),
        _point("2024-02-02T00:00:00Z", 23.1, 38.1),
    ])
    merged = merge_events([{"events": [early]}, {"events": [late]}])
    assert len(merged) == 1
    assert [g["date"] for g in merged[0]["geometry"]] == ["2024-01-30T00:00:00Z", "2024-02-02T00:00:00Z"]


def test_schema_rejects_reversed_dates(parsed):
    events, _ = parsed
    bad = events.reset_index()
    bad.loc[0, ["first_date", "last_date"]] = [pd.Timestamp("2024-09-10"), pd.Timestamp("2024-09-01")]
    with pytest.raises(SchemaError):
        EonetEvents.validate(bad)
