"""Re-aggregate REAL runs/ into test_results.json without re-running the LLM.

Walks ``<results_dir>/runs/`` and extracts step_confidence / step_u_request /
cum_reward from existing pickles + summary_info.json, then writes a
``test_results.json`` matching the format produced by proposed_v4_ya.py /
uprop.py (whichever fits given the agent type encoded in the run dir name).

Usage:
    uv run --group scripts python -m scripts.verification.real.reaggregate_runs \\
        --results_dir verification_results/real/react/glm-4.7
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)


def _aggregate(confs: list[float], method: str) -> float:
    if method == "last":
        return confs[-1]
    if method == "avg":
        return float(np.mean(confs))
    if method == "min":
        return float(np.min(confs))
    if method == "product":
        return float(np.exp(np.sum(np.log(np.clip(confs, 1e-9, 1.0)))))
    raise ValueError(method)


def _aggregate_u(scores: list[float], method: str) -> float:
    if method == "first":
        return scores[0]
    if method == "max":
        return float(np.max(scores))
    if method == "avg":
        return float(np.mean(scores))
    raise ValueError(method)


def _ece(confidences, labels, n_bins: int = 10) -> float:
    confs = np.array(confidences)
    labs = np.array(labels, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(confs)
    out = 0.0
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        out += mask.sum() * abs(confs[mask].mean() - labs[mask].mean())
    return float(out / n)


def _compute_conf_metrics(traj_confs, labels):
    if len(set(labels)) < 2:
        return {"auroc": 0.5, "ece": 1.0, "brier": 1.0}
    return {
        "auroc": float(roc_auc_score(labels, traj_confs)),
        "ece": _ece(traj_confs, labels),
        "brier": float(brier_score_loss(labels, traj_confs)),
    }


def aggregate_dir(results_dir: Path) -> dict:
    runs = results_dir / "runs"
    if not runs.is_dir():
        raise SystemExit(f"No runs/ dir under {results_dir}")

    task_details: dict = {}
    all_step_confidences: list[list[float]] = []
    all_step_u_requests: list[list[float]] = []
    labels: list[int] = []
    has_u_request = False
    skipped = 0

    for run_dir in sorted(runs.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary_info.json"
        if not summary_path.exists():
            skipped += 1
            continue
        try:
            summary = json.load(open(summary_path))
        except Exception:
            skipped += 1
            continue
        task_name = summary.get("task_name") or run_dir.name
        # If the same task_name appeared twice (e.g., multiple harness runs),
        # last writer wins, mirroring how the original aggregator behaves when
        # iterating over a results dict keyed by task_name.

        step_files = sorted(run_dir.glob("step_*.pkl.gz"))
        step_confidences: list[float] = []
        step_u_requests: list[float] = []
        for sf in step_files:
            try:
                with gzip.open(sf, "rb") as f:
                    sd = pickle.load(f)
            except Exception:
                continue
            ai = getattr(sd, "agent_info", None) or {}
            conf = ai.get("step_confidence") if isinstance(ai, dict) else None
            if conf is None:
                continue
            step_confidences.append(float(conf))
            u_req = ai.get("step_u_request") if isinstance(ai, dict) else None
            if u_req is not None:
                step_u_requests.append(float(u_req))
                has_u_request = True

        if not step_confidences:
            skipped += 1
            continue

        cum_reward = summary.get("cum_reward", 0) or 0
        success = 1 if cum_reward > 0 else 0

        all_step_confidences.append(step_confidences)
        if has_u_request:
            all_step_u_requests.append(step_u_requests or [0.0] * len(step_confidences))
        labels.append(success)

        detail = {
            "step_confidences": step_confidences,
            "success": success,
            "cum_reward": cum_reward,
            "task_agent_token_usage": {},
            "n_steps": len(step_files),
        }
        for m in ("last", "avg", "min", "product"):
            detail[f"c_{m}"] = _aggregate(step_confidences, m)
        if step_u_requests:
            detail["step_u_requests"] = step_u_requests
        task_details[task_name] = detail

    metrics: dict[str, dict[str, float]] = {}
    for m in ("last", "avg", "min", "product"):
        traj = [_aggregate(c, m) for c in all_step_confidences]
        metrics[f"confidence/{m}"] = _compute_conf_metrics(traj, labels)
    if has_u_request and all_step_u_requests:
        for m in ("first", "max", "avg"):
            traj_u = [_aggregate_u(u, m) for u in all_step_u_requests]
            inv = [1.0 - x for x in traj_u]
            metrics[f"u_request/{m}"] = _compute_conf_metrics(inv, labels)

    success_rate = sum(labels) / len(labels) if labels else 0.0

    out = {
        "model": "(reaggregated)",
        "primary_aggregation": "avg",
        "n_tasks": len(task_details),
        "max_steps": 25,
        "random_seed": 42,
        "success_rate": success_rate,
        "tasks_evaluated": len(task_details),
        "metrics": metrics,
        "task_details": task_details,
    }
    print(
        f"Aggregated {results_dir}: tasks={len(task_details)}, "
        f"success_rate={success_rate:.3f}, skipped={skipped}, "
        f"has_u_request={has_u_request}"
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_path = args.results_dir / "test_results.json"
    if out_path.exists() and not args.force:
        print(f"{out_path} already exists; pass --force to overwrite")
        return

    payload = aggregate_dir(args.results_dir)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
