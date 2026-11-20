"""Render a Markdown model card from run metadata and evaluation rows. Pure string building."""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone

DISCLAIMER = (
    "_EarthPulse is an exploratory research tool. Outputs are candidates for investigation, "
    "not warnings, and scores are not comparable across hazards._"
)


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def render(
    *,
    name: str,
    version: str,
    hazard: str,
    method: str,
    intended_use: str,
    not_for: str,
    data: dict,
    params: dict,
    evaluations: list[dict],
    limitations: list[str],
    region: str | None = None,
    train_start: date | None = None,
    train_end: date | None = None,
    mlflow_run_id: str | None = None,
) -> str:
    lines = [
        f"# Model card: `{name}` v{version}",
        "",
        f"- **Hazard:** {hazard}",
        f"- **Method:** {method}",
        f"- **Region:** {region or 'all configured regions'}",
        f"- **Training window:** {train_start or '—'} → {train_end or '—'}",
        f"- **MLflow run:** {mlflow_run_id or '—'}",
        f"- **Generated:** {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC",
        "",
        "## Intended use",
        intended_use,
        "",
        "## Not intended for",
        not_for,
        "",
        "## Data",
        *[f"- **{k}:** {v}" for k, v in data.items()],
        "",
        "## Parameters",
        "```json",
        json.dumps(params, indent=2, default=str),
        "```",
        "",
        "## Evaluation",
    ]
    if evaluations:
        lines += ["| metric | value | baseline | details |", "|---|---|---|---|"]
        for e in evaluations:
            lines.append(
                f"| {e.get('metric')} | {_fmt(e.get('value'))} | {e.get('baseline') or '—'} | "
                f"{json.dumps(e.get('details') or {}, default=str)} |"
            )
    else:
        lines.append("_No evaluation recorded yet._")
    lines += ["", "## Limitations", *[f"- {l}" for l in limitations], "", "---", DISCLAIMER]
    return "\n".join(lines) + "\n"