from sqlalchemy.dialects.postgresql import insert

from app.db import session_scope
from app.models import Source

SOURCES = [
    {
        "key": "eonet",
        "name": "NASA EONET v3",
        "url": "https://eonet.gsfc.nasa.gov/api/v3",
        "license": "NASA open data",
    },
    {
        "key": "firms_viirs_snpp",
        "name": "NASA FIRMS VIIRS S-NPP 375 m",
        "url": "https://firms.modaps.eosdis.nasa.gov/",
        "license": "NASA open data",
    },
    {
        "key": "gpm_imerg_daily",
        "name": "GPM IMERG Final Daily (GPM_3IMERGDF V07)",
        "url": "https://disc.gsfc.nasa.gov/datasets/GPM_3IMERGDF_07/summary",
        "license": "NASA open data",
    },
    {
        "key": "ibtracs",
        "name": "NOAA IBTrACS v04",
        "url": "https://www.ncei.noaa.gov/products/international-best-track-archive",
        "license": "NOAA public domain",
    },
    {
        "key": "gvp",
        "name": "Smithsonian GVP Volcanoes of the World",
        "url": "https://volcano.si.edu/",
        "license": "Free use with citation to the Global Volcanism Program",
    },
    {
        "key": "glc",
        "name": "NASA Global Landslide Catalog (COOLR)",
        "url": "https://gpm.nasa.gov/landslides/data.html",
        "license": "NASA open data",
    },
]


def seed_sources() -> int:
    stmt = insert(Source).values(SOURCES)
    stmt = stmt.on_conflict_do_update(
        index_elements=[Source.key],
        set_={"name": stmt.excluded.name, "url": stmt.excluded.url, "license": stmt.excluded.license},
    ).returning(Source.key)
    with session_scope() as session:
        return len(session.execute(stmt).all())


if __name__ == "__main__":
    print(f"sources upserted: {seed_sources()}")
