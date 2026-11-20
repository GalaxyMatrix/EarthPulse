import numpy as np 

EARTH_RADIUS_KM = 6371.0088

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    x = np.sin(lon2 - lon1) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def circular_mean_deg(angles_deg) -> float:
    a = np.radians(np.asarray(angles_deg, dtype=float))
    a = a[~np.isnan(a)]
    if a.size == 0:
        return float("nan")
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360)


def approx_area_km2(polygon) -> float:
    """Equirectangular approximation; adequate for small polygons such as fire-cluster hulls."""
    return float(polygon.area * (111.32**2) * np.cos(np.radians(polygon.centroid.y)))


def km_to_deg(km: float) -> float:
    return km / 111.32
