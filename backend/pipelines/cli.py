"""EarthPulse pipeline commands. Run from backend/:  python -m pipelines.cli --help"""
import logging
from datetime import date, datetime
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False, help="EarthPulse data pipelines.")
ingest_app = typer.Typer(no_args_is_help=True, help="Fetch raw data and load it into PostGIS.")
train_app = typer.Typer(no_args_is_help=True, help="Fit models and write model_runs.")
score_app = typer.Typer(no_args_is_help=True, help="Score data with a model run and persist results.")
eval_app = typer.Typer(no_args_is_help=True, help="Evaluate scored outputs against EONET.")
app.add_typer(ingest_app, name="ingest")
app.add_typer(train_app, name="train")
app.add_typer(score_app, name="score")
app.add_typer(eval_app, name="eval")

Start = Annotated[datetime, typer.Option(formats=["%Y-%m-%d"], help="First day (YYYY-MM-DD).")]
End = Annotated[datetime | None, typer.Option(formats=["%Y-%m-%d"], help="Last day (YYYY-MM-DD); default today.")]
Region = Annotated[str, typer.Option(help="Region key from config/regions.yaml.")]
RunId = Annotated[int, typer.Option(help="model_runs.id")]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _days(start: datetime, end: datetime | None) -> tuple[date, date]:
    first, last = start.date(), (end or datetime.now()).date()
    if first > last:
        raise typer.BadParameter(f"--start {first} is after --end {last}")
    return first, last


def _report(run) -> None:
    typer.echo(
        f"ingest run {run.id}: {run.status}, {run.rows_in} rows in, {run.rows_written} written; raw: {run.raw_path}"
    )


@app.command("seed-sources")
def seed_sources() -> None:
    """Insert or update the rows in `sources`."""
    from pipelines.seed_sources import seed_sources as run

    typer.echo(f"sources upserted: {run()}")


@app.command("rain-cells")
def rain_cells(region: Region) -> None:
    """Create the region's IMERG 0.1° cells in `rain_cells`."""
    from pipelines.ingest.gpm import ensure_cells

    typer.echo(f"{region}: {ensure_cells(region)} rain cells upserted")


@app.command("model-card")
def model_card(run_id: RunId) -> None:
    """Render docs/model_cards/<name>.md from a model run and its evaluations."""
    from pipelines.model_card import write_card

    path = write_card(run_id)
    typer.echo(f"wrote {path}")


@ingest_app.command("eonet")
def ingest_eonet(
    start: Start,
    end: End = None,
    region: Annotated[str | None, typer.Option(help="Limit to this region's bbox (default: global).")] = None,
) -> None:
    """EONET events (all categories, open and closed)."""
    from pipelines.common import get_region
    from pipelines.ingest import eonet

    first, last = _days(start, end)
    _report(eonet.ingest(first, last, bbox=get_region(region)["bbox"] if region else None))


@ingest_app.command("firms")
def ingest_firms(region: Region, start: Start, end: End = None) -> None:
    """FIRMS VIIRS S-NPP detections for the region's bbox."""
    from pipelines.ingest import firms

    first, last = _days(start, end)
    _report(firms.ingest(region, first, last))


@ingest_app.command("rain")
def ingest_rain(
    region: Region,
    start: Start,
    end: End = None,
    force: Annotated[bool, typer.Option(help="Re-download days that are already loaded.")] = False,
) -> None:
    """GPM IMERG Final daily precipitation for the region's rain cells."""
    from pipelines.ingest import gpm

    first, last = _days(start, end)
    _report(gpm.ingest(region, first, last, force=force))


@train_app.command("fire-anomaly")
def train_fire_anomaly(region: Region, start: Start, end: End = None) -> None:
    """Fit Isolation Forest on VIIRS detections in [start, end]."""
    from pipelines.train import fire_anomaly

    first, last = _days(start, end)
    run_id = fire_anomaly.train(region, first, last)
    typer.echo(f"model run {run_id}: fire_anomaly_if")


@train_app.command("rain-climatology")
def train_rain_climatology(
    region: Region,
    start: Start,
    end: End = None,
    min_years: Annotated[int, typer.Option(help="Min calendar years per cell (use 1 for short demos).")] = 3,
) -> None:
    """Build per-cell DOY climatology from rain_daily."""
    from pipelines.train import rain_climatology

    first, last = _days(start, end)
    run_id = rain_climatology.train(region, first, last, min_years=min_years)
    typer.echo(f"model run {run_id}: rain_climatology")


@train_app.command("rain-forecast")
def train_rain_forecast(
    clim_run_id: Annotated[int, typer.Option(help="rain_climatology model_runs.id")],
    start: Start,
    train_end: Annotated[datetime, typer.Option(formats=["%Y-%m-%d"], help="Last train target date.")],
    end: End = None,
) -> None:
    """Baselines + LightGBM next-day forecast; writes skill metrics on the LGBM run."""
    from pipelines.train import rain_forecast

    first, last = _days(start, end)
    ids = rain_forecast.train(clim_run_id, start=first, end=last, train_end=train_end.date())
    typer.echo(f"forecast runs: {ids}")


@score_app.command("fire-anomaly")
def score_fire_anomaly(run_id: RunId, start: Start, end: End = None) -> None:
    """Score detections with a trained fire_anomaly_if run."""
    from pipelines.score import fire_anomaly

    first, last = _days(start, end)
    n = fire_anomaly.score(run_id, first, last)
    typer.echo(f"scored {n} detections for run {run_id}")


@score_app.command("fire-clusters")
def score_fire_clusters(region: Region, start: Start, end: End = None) -> None:
    """Daily DBSCAN clusters for the region's detections."""
    from pipelines.score import fire_clusters

    first, last = _days(start, end)
    run_id = fire_clusters.score(region, first, last)
    typer.echo(f"model run {run_id}: fire_clusters_dbscan")


@score_app.command("rain-anomaly")
def score_rain_anomaly(
    clim_run_id: Annotated[int, typer.Option(help="rain_climatology model_runs.id")],
    start: Start,
    end: End = None,
    window_days: Annotated[int, typer.Option(help="Which rolling window to store (DB allows one).")] = 7,
) -> None:
    """Z-score anomalies vs climatology; creates a rain_anomaly_z model run."""
    from pipelines.score import rain_anomaly

    first, last = _days(start, end)
    run_id = rain_anomaly.score(clim_run_id, first, last, window_days=window_days)
    typer.echo(f"model run {run_id}: rain_anomaly_z")


@eval_app.command("fire-anomaly")
def eval_fire_anomaly(run_id: RunId, start: Start, end: End = None) -> None:
    """Precision@k / hit-rate vs EONET wildfires for scored detections."""
    from app.db import session_scope
    from pipelines.common import get_region
    from pipelines.evaluate import fire_vs_eonet
    from app.models import ModelRun

    first, last = _days(start, end)
    with session_scope() as session:
        run = session.get(ModelRun, run_id)
        bbox = get_region(run.region)["bbox"]
        n = fire_vs_eonet.run_anomaly_eval(session, run_id=run_id, bbox=bbox, start=first, end=last)
    typer.echo(f"wrote {n} evaluation rows for run {run_id}")


@eval_app.command("fire-clusters")
def eval_fire_clusters(run_id: RunId, start: Start, end: End = None) -> None:
    """Cluster coverage vs EONET wildfires."""
    from app.db import session_scope
    from app.models import ModelRun
    from pipelines.common import get_region
    from pipelines.evaluate import fire_vs_eonet

    first, last = _days(start, end)
    with session_scope() as session:
        run = session.get(ModelRun, run_id)
        bbox = get_region(run.region)["bbox"]
        n = fire_vs_eonet.run_cluster_eval(session, run_id=run_id, bbox=bbox, start=first, end=last)
    typer.echo(f"wrote {n} evaluation rows for run {run_id}")


@eval_app.command("rain-anomaly")
def eval_rain_anomaly(
    run_id: RunId,
    start: Start,
    end: End = None,
    window_days: Annotated[int, typer.Option()] = 7,
) -> None:
    """Precision@k vs EONET floods for rain anomalies."""
    from app.db import session_scope
    from app.models import ModelRun
    from pipelines.common import get_region
    from pipelines.evaluate import rain_vs_eonet
    from pipelines.evaluate.rain_vs_eonet import RainEvalParams

    first, last = _days(start, end)
    with session_scope() as session:
        run = session.get(ModelRun, run_id)
        # EONET floods near Attica are sparse; score cells stay regional, match search is wider
        bbox = get_region(run.region)["bbox"]
        wide = (min(19.0, bbox[0]), min(34.0, bbox[1]), max(30.0, bbox[2]), max(42.0, bbox[3]))
        n = rain_vs_eonet.run_eval(
            session, run_id=run_id, bbox=wide, start=first, end=last,
            params=RainEvalParams(window_days=window_days),
        )
    typer.echo(f"wrote {n} evaluation rows for run {run_id}")


if __name__ == "__main__":
    app()
