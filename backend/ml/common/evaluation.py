from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from ml.common.geo import EARTH_RADIUS_KM



@dataclass(frozen=True)
class MatchParams:
    buffer_km: float = 25.0 
    before_days: int = 2
    after_days: int = 2


def match_to_events(candidates: pd.DataFrame, events: pd.DataFrame, params: MatchParams = MatchParams()) -> pd.DataFrame:
    out = candidates.copy()
    out["matched_event_id"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
    out["match_dist_km"] = np.nan
    out["lead_days"] = np.nan
    if candidates.empty or events.empty:
        return out 
    

    tree = BallTree(np.radians(events[["lat", "lon"]].to_numpy()), metric="haversine")
    idx_lists, dist_lists = tree.query_radius(
        np.radians(candidates[["lat", "lon"]].to_numpy()),
        r=params.buffer_km / EARTH_RADIUS_KM,
        return_distance=True,
        sort_results=True,
    )
    cdate = pd.to_datetime(candidates["date"]).to_numpy()
    first = pd.to_datetime(events["first_date"]).to_numpy()
    last = pd.to_datetime(events["last_date"]).to_numpy()
    event_ids = events["event_id"].to_numpy()
    before = np.timedelta64(params.before_days, "D")
    after = np.timedelta64(params.after_days, "D") 

    matched, dist, lead = [pd.NA] * len(out), np.full(len(out), np.nan), np.full(len(out), np.nan)
    for i, (idxs, dists) in enumerate(zip(idx_lists, dist_lists)):
        for j, d_rad in zip(idxs, dists):
            if first[j] - before <= cdate[i] <= last[j] + after:
                matched[i] = event_ids[j]
                dist[i] = d_rad * EARTH_RADIUS_KM
                lead[i] = (cdate[i] - first[j]) / np.timedelta64(1, "D")
                break
    out["matched_event_id"] = matched
    out["match_dist_km"] = dist
    out["lead_days"] = lead
    return out
def precision_at_k(matched: pd.DataFrame, k: int, score_col: str = "score") -> float:
    top = matched.nlargest(k, score_col)
    return float(top["matched_event_id"].notna().mean()) if len(top) else float("nan")
def hit_rate(matched: pd.DataFrame, events: pd.DataFrame, k: int | None = None, score_col: str = "score") -> float:
    """Fraction of catalogue events with at least one (top-k) candidate matched to them."""
    if events.empty:
        return float("nan")
    pool = matched.nlargest(k, score_col) if k else matched
    hits = set(pool["matched_event_id"].dropna()) & set(events["event_id"])
    return float(len(hits) / len(events))
def random_precision(matched: pd.DataFrame) -> float:
    """Expected precision@k of a random ranking: the base match rate. Every precision@k is reported against this."""
    return float(matched["matched_event_id"].notna().mean()) if len(matched) else float("nan")
def retrieval_report(matched: pd.DataFrame, events: pd.DataFrame, ks=(10, 50, 100, 500), score_col: str = "score") -> list[dict]:
    """Rows shaped for the `evaluations` table / MLflow metrics."""
    base = random_precision(matched)
    rows = [{"metric": "random_precision", "value": base, "baseline": None, "details": {"n": int(len(matched))}}]
    for k in ks:
        rows.append({"metric": f"precision_at_{k}", "value": precision_at_k(matched, k, score_col), "baseline": "random_precision", "details": {"k": k}})
        rows.append({"metric": f"hit_rate_at_{k}", "value": hit_rate(matched, events, k, score_col), "baseline": None, "details": {"k": k}})
    rows.append({"metric": "hit_rate", "value": hit_rate(matched, events, None, score_col), "baseline": None, "details": {}})
    leads = matched["lead_days"].dropna()
    if len(leads):
        rows.append({"metric": "median_lead_days", "value": float(leads.median()), "baseline": None, "details": {"n": int(len(leads))}})
        rows.append({"metric": "frac_before_event", "value": float((leads < 0).mean()), "baseline": None, "details": {"n": int(len(leads))}})
    return rows
def cluster_coverage_report(matched_clusters: pd.DataFrame, events: pd.DataFrame) -> list[dict]:
    """For clustering (DBSCAN) outputs matched with match_to_events on cluster centroids."""
    return [
        {"metric": "cluster_match_rate", "value": random_precision(matched_clusters), "baseline": None, "details": {"n_clusters": int(len(matched_clusters))}},
        {"metric": "event_coverage", "value": hit_rate(matched_clusters, events, None, score_col="n_detections"), "baseline": None, "details": {"n_events": int(len(events))}},
    ]
def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))
def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2)))
def skill_score(y, p, p_baseline) -> float:
    """1 - MSE_model / MSE_baseline. > 0 beats the baseline, 0 ties, < 0 is worse."""
    y = np.asarray(y, float)
    mse_m = np.mean((y - np.asarray(p, float)) ** 2)
    mse_b = np.mean((y - np.asarray(p_baseline, float)) ** 2)
    return float(1 - mse_m / mse_b) if mse_b > 0 else float("nan")
def wet_day_scores(y, p, threshold_mm: float = 1.0) -> dict:
    """Categorical skill for 'rain / no rain' at a threshold: hit rate, false alarm ratio, CSI."""
    obs, fc = np.asarray(y) >= threshold_mm, np.asarray(p) >= threshold_mm
    hits, misses, fa = int((obs & fc).sum()), int((obs & ~fc).sum()), int((~obs & fc).sum())
    return {
        "pod": hits / (hits + misses) if hits + misses else float("nan"),
        "far": fa / (hits + fa) if hits + fa else float("nan"),
        "csi": hits / (hits + misses + fa) if hits + misses + fa else float("nan"),
    }
def forecast_report(y, preds: dict[str, np.ndarray], baseline: str) -> list[dict]:
    """preds: {model_name: predictions}. MAE/RMSE/CSI for each, skill vs the named baseline."""
    rows = []
    for name, p in preds.items():
        rows.append({"model": name, "metric": "mae", "value": mae(y, p), "baseline": None, "details": {}})
        rows.append({"model": name, "metric": "rmse", "value": rmse(y, p), "baseline": None, "details": {}})
        rows.append({"model": name, "metric": "csi_1mm", "value": wet_day_scores(y, p)["csi"], "baseline": None, "details": {"threshold_mm": 1.0}})
        if name != baseline and baseline in preds:
            rows.append({"model": name, "metric": "skill_vs_baseline", "value": skill_score(y, p, preds[baseline]), "baseline": baseline, "details": {}})
    return rows
def params_dict(params: MatchParams) -> dict:
    return asdict(params)



