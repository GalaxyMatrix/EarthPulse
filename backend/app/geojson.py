"""Build GeoJSON FeatureCollections from PostGIS ST_AsGeoJSON rows."""
from __future__ import annotations

import json
import math
from typing import Any


def feature(geometry_json: str | dict | None, properties: dict[str, Any]) -> dict:
    geom = json.loads(geometry_json) if isinstance(geometry_json, str) else geometry_json
    return {"type": "Feature", "geometry": geom, "properties": properties}


def feature_collection(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}


def jsonable(value: Any) -> Any:
    """Make SQLAlchemy/pandas scalars safe for JSON (dates, decimals, etc.)."""
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
