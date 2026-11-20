from fastapi import FastAPI

from app.routers import events, fires, health, model_runs, rain

app = FastAPI(
    title="EarthPulse",
    version="0.1.0",
    description="Hazard map layers and model metadata as GeoJSON / JSON.",
)

app.include_router(health.router)
app.include_router(events.router, prefix="/events", tags=["events"])
app.include_router(fires.router, tags=["fires"])
app.include_router(rain.router, prefix="/rain", tags=["rain"])
app.include_router(model_runs.router, prefix="/model-runs", tags=["ml"])
