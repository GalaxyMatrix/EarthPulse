"""API smoke tests against the live PostGIS DB (Attica demo data)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.db import engine
from app.main import app

BBOX = "22.8,37.6,24.6,39.1"
CLIENT = TestClient(app)


def _db_up() -> bool:
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_up(), reason="PostGIS not reachable")


def test_health_ok():
    r = CLIENT.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_bbox_validation():
    r = CLIENT.get("/events", params={"bbox": "1,2,3", "start": "2025-01-01", "end": "2025-01-02"})
    assert r.status_code == 422


def _assert_fc(data: dict):
    assert data["type"] == "FeatureCollection"
    assert isinstance(data["features"], list)
    for f in data["features"][:5]:
        assert f["type"] == "Feature"
        assert "geometry" in f
        assert "properties" in f


def test_events_geojson():
    r = CLIENT.get(
        "/events",
        params={"bbox": BBOX, "start": "2025-07-01", "end": "2026-09-30", "category": "wildfires"},
    )
    assert r.status_code == 200
    _assert_fc(r.json())


def test_fires_geojson():
    r = CLIENT.get(
        "/fires",
        params={"bbox": BBOX, "start": "2026-07-01", "end": "2026-09-30"},
    )
    assert r.status_code == 200
    body = r.json()
    _assert_fc(body)
    assert len(body["features"]) > 0


def test_fire_clusters_geojson():
    r = CLIENT.get(
        "/fire-clusters",
        params={"bbox": BBOX, "start": "2026-07-01", "end": "2026-09-30"},
    )
    assert r.status_code == 200
    _assert_fc(r.json())


def test_rain_daily_geojson():
    r = CLIENT.get(
        "/rain/daily",
        params={"bbox": "23.2,37.8,24.2,38.5", "start": "2024-01-01", "end": "2024-01-07", "region": "attica"},
    )
    assert r.status_code == 200
    body = r.json()
    _assert_fc(body)
    assert len(body["features"]) > 0


def test_rain_anomalies_geojson():
    r = CLIENT.get(
        "/rain/anomalies",
        params={"bbox": "23.2,37.8,24.2,38.5", "start": "2023-11-01", "end": "2024-01-25"},
    )
    assert r.status_code == 200
    _assert_fc(r.json())


def test_model_runs_and_evaluations():
    r = CLIENT.get("/model-runs", params={"hazard": "wildfire"})
    assert r.status_code == 200
    runs = r.json()
    assert isinstance(runs, list)
    if not runs:
        pytest.skip("no model runs")
    run_id = runs[0]["id"]
    detail = CLIENT.get(f"/model-runs/{run_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == run_id
    ev = CLIENT.get(f"/model-runs/{run_id}/evaluations")
    assert ev.status_code == 200
    assert isinstance(ev.json(), list)
