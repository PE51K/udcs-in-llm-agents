#!/usr/bin/env python3
"""AUQ ROC-AUC Verification on REAL Benchmark
===========================================

Verifies AUQ (Agentic Uncertainty Quantification) from Huang et al.
The agent elicits verbalized confidence via <confidence> tags; trajectory-level
confidence is aggregated from per-step values using four methods (Section 3.4):

    Φ_last    (End-State Belief):    C(τ) = ĉ_T
    Φ_avg     (Overall Quality):     C(τ) = (1/T) Σ ĉ_t
    Φ_min     (Process Reliability): C(τ) = min ĉ_t
    Φ_product (Chain Probability):   C(τ) = Π ĉ_t

For each aggregation three metrics are computed:
    AUROC       - discrimination ability (high confidence → success)
    ECE         - calibration error (T-ECE from paper)
    Brier Score - mean squared error between confidence and outcome

Run from project root:
    # Requires .env with OPENAI_API_KEY (and optional MODEL)
    uv run --group scripts python -m \
        scripts.verification.real.proposed_v4.proposed_v4 \
        --n_tasks 100 --log_level DEBUG --random_seed 42

Arguments:
  --model               LLM model name (default: openrouter/openai/gpt-5.1)
  --primary_aggregation Method highlighted in logs: last|avg|min|product (default: avg)
  --n_tasks             Number of REAL tasks to run (default: 100)
  --max_steps           Maximum steps per episode (default: 25)
  --results_dir         Output directory (default: ./verification_results/real/proposed_v4)
  --log_level           Logging level: DEBUG|INFO|WARNING|ERROR|CRITICAL (default: INFO)
  --random_seed             Random seed for reproducibility (default: None)
  --shuffle_tasks           Randomly shuffle tasks before selection (default: True)
  --include_obs_in_history  Include observation summaries in UAM history (default: False)
  --no-include_obs_in_history  Disable observations in UAM history

Output:
  test_results.json with per-method metrics (auroc/ece/brier) and per-task details
  (step_confidences, c_last, c_avg, c_min, c_product, success, cum_reward)
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
import dataclasses
import gzip
import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Any, Literal

import numpy as np
from agisdk import REAL
from agisdk.REAL.browsergym.experiments import AbstractAgentArgs
from sklearn.metrics import roc_auc_score, brier_score_loss

from .hackable import ProposedV4HackableAgent

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ProposedV4AgentArgs(AbstractAgentArgs):
    """Agent args for proposed v4: UAM confidence + u_request."""

    agent_name: str = "ProposedV4Agent"
    model_name: str = "openrouter/openai/gpt-5.1"
    provider: str | None = None
    chat_mode: bool = False
    demo_mode: str = "off"
    use_html: bool = False
    use_axtree: bool = True
    use_screenshot: bool = True
    temperature: float = 0.0
    include_obs_in_history: bool = False
    system_message_handling: Literal["separate", "combined"] = "separate"

    def make_agent(self) -> ProposedV4HackableAgent:
        return ProposedV4HackableAgent(
            model_name=self.model_name,
            provider=self.provider,
            chat_mode=self.chat_mode,
            demo_mode=self.demo_mode,
            use_html=self.use_html,
            use_axtree=self.use_axtree,
            use_screenshot=self.use_screenshot,
            temperature=self.temperature,
            include_obs_in_history=self.include_obs_in_history,
            system_message_handling=self.system_message_handling,
        )


class TestRunner:
    """Runs test on REAL benchmark tasks."""

    DEFAULT_TASKS = [
        # GoCalendar tasks (calendar/scheduling)
        "v2.gocalendar-4",
        "v2.gocalendar-6",
        "v2.gocalendar-8",
        "v2.gocalendar-10",
        "v2.gocalendar-11",
        "v2.gocalendar-12",
        "v2.gocalendar-13",
        "v2.gocalendar-16",
        "v2.gocalendar-18",
        "v2.gocalendar-23",
        # GoMail tasks (email)
        "v2.gomail-1",
        "v2.gomail-2",
        "v2.gomail-4",
        "v2.gomail-5",
        "v2.gomail-7",
        "v2.gomail-8",
        "v2.gomail-9",
        "v2.gomail-11",
        "v2.gomail-12",
        "v2.gomail-13",
        "v2.gomail-14",
        "v2.gomail-15",
        "v2.gomail-16",
        "v2.gomail-17",
        "v2.gomail-18",
        "v2.gomail-19",
        "v2.gomail-20",
        "v2.gomail-21",
        "v2.gomail-23",
        "v2.gomail-24",
        "v2.gomail-26",
        # NetworkIn tasks (social network)
        "v2.networkin-1",
        "v2.networkin-5",
        "v2.networkin-6",
        "v2.networkin-10",
        "v2.networkin-11",
        "v2.networkin-13",
        "v2.networkin-19",
        "v2.networkin-25",
        "v2.networkin-26",
        "v2.networkin-27",
        # OpenDining tasks (restaurant reservations)
        "v2.opendining-1",
        "v2.opendining-2",
        "v2.opendining-4",
        "v2.opendining-5",
        "v2.opendining-6",
        "v2.opendining-7",
        "v2.opendining-9",
        "v2.opendining-12",
        "v2.opendining-17",
        "v2.opendining-20",
        "v2.opendining-24",
        # StayNB tasks (vacation rentals)
        "v2.staynb-1",
        "v2.staynb-4",
        "v2.staynb-5",
        "v2.staynb-6",
        "v2.staynb-9",
        "v2.staynb-12",
        "v2.staynb-13",
        "v2.staynb-14",
        "v2.staynb-15",
        "v2.staynb-17",
        "v2.staynb-19",
        # TopWork tasks (job board)
        "v2.topwork-2",
        "v2.topwork-3",
        "v2.topwork-4",
        "v2.topwork-7",
        "v2.topwork-8",
        "v2.topwork-9",
        "v2.topwork-10",
        "v2.topwork-11",
        "v2.topwork-12",
        "v2.topwork-13",
        "v2.topwork-14",
        "v2.topwork-21",
        # UDriver tasks (ride sharing)
        "v2.udriver-2",
        "v2.udriver-3",
        "v2.udriver-4",
        "v2.udriver-6",
        "v2.udriver-7",
        "v2.udriver-9",
        "v2.udriver-11",
        "v2.udriver-13",
        "v2.udriver-14",
        "v2.udriver-15",
        "v2.udriver-16",
        "v2.udriver-17",
        "v2.udriver-19",
        "v2.udriver-20",
        # Zilloft tasks (real estate)
        "v2.zilloft-2",
        "v2.zilloft-3",
        "v2.zilloft-5",
        "v2.zilloft-6",
        "v2.zilloft-9",
        "v2.zilloft-10",
        "v2.zilloft-14",
        "v2.zilloft-15",
        "v2.zilloft-22",
        "v2.zilloft-23",
        # MarriSuite tasks (hotel)
        "v2.marrisuite-8",
    ]

    def __init__(
        self,
        model_name: str = "openrouter/openai/gpt-5.1",
        provider: str | None = None,
        include_obs_in_history: bool = False,
        use_screenshot: bool = True,
        results_dir: str = "./verification_results",
        task_subset: list[str] | None = None,
        max_steps: int = 25,
        shuffle_tasks: bool = True,
    ):
        self.model_name = model_name
        self.provider = provider
        self.include_obs_in_history = include_obs_in_history
        self.use_screenshot = use_screenshot
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.task_subset = task_subset or self.DEFAULT_TASKS.copy()

        if shuffle_tasks:
            random.shuffle(self.task_subset)
            logger.info(f"Test runner: {len(self.task_subset)} tasks (shuffled)")
        else:
            logger.info(f"Test runner: {len(self.task_subset)} tasks")

    def run_agent(self) -> dict[str, dict[str, Any]]:
        """Run agent with configured parameters on all tasks."""
        logger.info(f"Running proposed v4 agent: model={self.model_name}")

        agent_args = ProposedV4AgentArgs(
            agent_name="ProposedV4Agent",
            model_name=self.model_name,
            provider=self.provider,
            use_screenshot=self.use_screenshot,
            include_obs_in_history=self.include_obs_in_history,
        )

        harness = REAL.harness(
            agentargs=agent_args,
            task_name=None,
            headless=True,
            max_steps=self.max_steps,
            use_axtree=True,
            use_screenshot=self.use_screenshot,
            results_dir=str(self.results_dir / "runs"),
            use_cache=True,
            save_step_info_pkl=True,
        )

        logger.info(f"Starting harness run on {len(self.task_subset)} tasks")
        results = harness.run(tasks=self.task_subset)
        logger.info(f"Harness run completed, got {len(results)} results")

        return results

    def extract_episode_data(
        self, results: dict[str, dict]
    ) -> tuple[list[list[float]], list[list[float]], list[int], dict]:
        """Extract per-step confidences, u_requests, and success labels from results.

        Returns:
            all_step_confidences: Per-step confidence lists for each episode
            all_step_u_requests: Per-step u_request lists for each episode
            labels: Binary success labels (1=success, 0=failure)
            task_details: Per-task detail dict
        """
        all_step_confidences: list[list[float]] = []
        all_step_u_requests: list[list[float]] = []
        labels: list[int] = []
        task_details: dict = {}
        logger.info(f"Extracting episode data from {len(results)} results")

        for task_name, result in results.items():
            exp_dir = Path(result.get("exp_dir", ""))
            if not exp_dir.exists():
                logger.warning(f"Missing exp_dir: {task_name}")
                continue

            try:
                step_files = sorted(exp_dir.glob("step_*.pkl.gz"))
                if not step_files:
                    logger.debug(f"No step files for {task_name}")
                    continue

                step_confidences: list[float] = []
                step_u_requests: list[float] = []
                for step_file in step_files:
                    with gzip.open(step_file, "rb") as f:
                        step_data = pickle.load(f)
                    conf = step_data.agent_info.get("step_confidence")
                    u_req = step_data.agent_info.get("step_u_request", 0.0)
                    if conf is not None:
                        step_confidences.append(float(conf))
                        step_u_requests.append(float(u_req))

                if not step_confidences:
                    logger.warning(f"  {task_name}: no step_confidence values found")
                    continue

                success = 1 if result.get("cum_reward", 0) > 0 else 0
                all_step_confidences.append(step_confidences)
                all_step_u_requests.append(step_u_requests)
                labels.append(success)

                # Collect token usage from the last step
                task_agent_tokens: dict = {}
                with gzip.open(step_files[-1], "rb") as f:
                    last_step_data = pickle.load(f)
                task_agent_tokens = last_step_data.agent_info.get(
                    "task_agent_token_usage", {}
                )

                task_details[task_name] = {
                    "step_confidences": step_confidences,
                    "step_u_requests": step_u_requests,
                    "success": success,
                    "cum_reward": result.get("cum_reward", 0),
                    "task_agent_token_usage": task_agent_tokens,
                    "n_steps": len(step_files),
                }
                logger.info(
                    f"  {task_name}: steps={len(step_confidences)}, success={success}, "
                    f"agent_tokens={task_agent_tokens.get('total_tokens', '?')}"
                )

            except Exception as e:
                logger.error(f"Error processing {task_name}: {e}")

        logger.info(f"Extracted {len(labels)} episodes")
        return all_step_confidences, all_step_u_requests, labels, task_details

    @staticmethod
    def _aggregate(confs: list[float], method: str) -> float:
        if method == "last":
            return confs[-1]
        if method == "avg":
            return float(np.mean(confs))
        if method == "min":
            return float(np.min(confs))
        if method == "product":
            return float(np.exp(np.sum(np.log(np.clip(confs, 1e-9, 1.0)))))
        raise ValueError(f"Unknown aggregation method: {method!r}")

    @staticmethod
    def _aggregate_u_request(scores: list[float], method: str) -> float:
        if method == "first":
            return scores[0]
        if method == "max":
            return float(np.max(scores))
        if method == "avg":
            return float(np.mean(scores))
        if method == "product":
            # plain product over (1 - u), mirroring the confidence product
            return float(1.0 - np.exp(np.sum(np.log(np.clip([1.0 - s for s in scores], 1e-9, 1.0)))))
        raise ValueError(f"Unknown u_request aggregation method: {method!r}")

    @staticmethod
    def _ece(confidences: list[float], labels: list[int], n_bins: int = 10) -> float:
        """Expected Calibration Error (T-ECE from paper Section 3.4).

        T-ECE = Σ (|B_m|/N) * |acc(B_m) - conf(B_m)|
        Last bin is right-inclusive [0.9, 1.0] so confidence=1.0 is counted.
        """
        confs = np.array(confidences)
        labs = np.array(labels, dtype=float)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        n = len(confs)
        ece = 0.0
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            if i == n_bins - 1:
                mask = (confs >= lo) & (confs <= hi)
            else:
                mask = (confs >= lo) & (confs < hi)
            if mask.sum() == 0:
                continue
            ece += mask.sum() * abs(confs[mask].mean() - labs[mask].mean())
        return float(ece / n)

    @staticmethod
    def compute_metrics(
        all_step_confidences: list[list[float]],
        all_step_u_requests: list[list[float]],
        labels: list[int],
    ) -> dict[str, dict[str, float]]:
        """Compute AUROC, ECE, Brier for confidence aggregations and u_request aggregations.

        Confidence aggregations: last, avg, min, product
        u_request aggregations: first, max, avg (inverted for ROC-AUC)

        Returns dict keyed by "confidence/{method}" and "u_request/{method}".
        """
        fallback = {"auroc": 0.5, "ece": 1.0, "brier": 1.0}
        if len(set(labels)) < 2:
            logger.warning("Only one class present, returning fallback metrics")
            keys = [f"confidence/{m}" for m in ("last", "avg", "min", "product")]
            keys += [f"u_request/{m}" for m in ("first", "max", "avg", "product")]
            return {k: fallback.copy() for k in keys}

        results: dict[str, dict[str, float]] = {}

        # Confidence aggregations
        for method in ("last", "avg", "min", "product"):
            traj_confs = [
                TestRunner._aggregate(confs, method) for confs in all_step_confidences
            ]
            try:
                auroc = roc_auc_score(labels, traj_confs)
                ece = TestRunner._ece(traj_confs, labels)
                brier = brier_score_loss(labels, traj_confs)
                results[f"confidence/{method}"] = {"auroc": auroc, "ece": ece, "brier": brier}
                logger.info(
                    f"  confidence/{method}: AUROC={auroc:.4f}, ECE={ece:.4f}, Brier={brier:.4f}"
                )
            except Exception as e:
                logger.error(f"Error computing metrics for confidence/{method}: {e}")
                results[f"confidence/{method}"] = fallback.copy()

        # u_request aggregations (inverted: high u_request = low confidence in success)
        for method in ("first", "max", "avg", "product"):
            traj_u_reqs = [
                TestRunner._aggregate_u_request(reqs, method) for reqs in all_step_u_requests
            ]
            traj_confs_from_u = [1.0 - u for u in traj_u_reqs]
            try:
                auroc = roc_auc_score(labels, traj_confs_from_u)
                ece = TestRunner._ece(traj_confs_from_u, labels)
                brier = brier_score_loss(labels, traj_confs_from_u)
                results[f"u_request/{method}"] = {"auroc": auroc, "ece": ece, "brier": brier}
                logger.info(
                    f"  u_request/{method}: AUROC={auroc:.4f}, ECE={ece:.4f}, Brier={brier:.4f}"
                )
            except Exception as e:
                logger.error(f"Error computing metrics for u_request/{method}: {e}")
                results[f"u_request/{method}"] = fallback.copy()

        return results


def main():

    parser = argparse.ArgumentParser(description="AUQ verification on REAL")
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL", "openrouter/openai/gpt-5.1"),
        help="LLM model name",
    )
    parser.add_argument(
        "--primary_aggregation",
        default="avg",
        choices=["last", "avg", "min", "product"],
        help="Primary aggregation method highlighted in logs",
    )
    parser.add_argument("--n_tasks", type=int, default=100, help="Number of REAL tasks to run")
    parser.add_argument("--max_steps", type=int, default=25, help="Maximum steps per episode")
    parser.add_argument(
        "--results_dir",
        default=None,
        help="Output directory (default: ./verification_results/real/proposed_v4/{model})",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--shuffle_tasks", type=bool, default=True)
    parser.add_argument(
        "--include_obs_in_history",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include observation summaries in UAM history (paper Variant B)",
    )
    parser.add_argument(
        "--use_screenshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send the page screenshot to the model. Use --no-use_screenshot "
        "for non-multimodal backbones (axtree-only).",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("PROVIDER"),
        help="OpenRouter provider slug(s) to pin, comma-separated for an ordered allowlist (e.g. DeepInfra,Parasail); disables fallback. Default none (auto-route).",
    )
    args = parser.parse_args()

    if args.results_dir is None:
        model_dir = args.model.removeprefix("openrouter/").removeprefix("openai/").replace("/", "--")
        args.results_dir = f"./verification_results/real/proposed_v4/{model_dir}"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    log_level = getattr(logging, args.log_level)
    logging.getLogger("__main__").setLevel(log_level)
    logging.getLogger("scripts").setLevel(log_level)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        logger.info(f"Random seed set to: {args.random_seed}")

    logger.info("=" * 60)
    logger.info("AUQ Verification on REAL")
    logger.info(
        f"Model: {args.model}, Tasks: {args.n_tasks}, "
        f"obs_in_history: {args.include_obs_in_history}"
    )
    logger.info("=" * 60)

    runner = TestRunner(
        model_name=args.model,
        provider=args.provider,
        include_obs_in_history=args.include_obs_in_history,
        use_screenshot=args.use_screenshot,
        results_dir=args.results_dir,
        max_steps=args.max_steps,
        shuffle_tasks=args.shuffle_tasks,
    )

    if args.n_tasks < len(runner.task_subset):
        runner.task_subset = runner.task_subset[: args.n_tasks]

    results = runner.run_agent()
    all_step_confidences, all_step_u_requests, labels, task_details = runner.extract_episode_data(results)

    if not labels:
        logger.error("No confidence values extracted, cannot compute metrics")
        return

    metrics = runner.compute_metrics(all_step_confidences, all_step_u_requests, labels)
    success_rate = float(np.mean(labels))

    logger.info("=" * 60)
    logger.info("Test Results:")
    for method, m in metrics.items():
        marker = " ◄" if method == args.primary_aggregation else ""
        logger.info(
            f"  Φ_{method:<8}: AUROC={m['auroc']:.4f}  ECE={m['ece']:.4f}  Brier={m['brier']:.4f}{marker}"
        )
    logger.info(f"  Success Rate: {success_rate:.2%}")
    logger.info(f"  Tasks evaluated: {len(labels)}")
    logger.info("=" * 60)

    # Annotate task_details with all four trajectory-level confidences
    for detail in task_details.values():
        confs = detail["step_confidences"]
        for method in ("last", "avg", "min", "product"):
            detail[f"c_{method}"] = TestRunner._aggregate(confs, method)

    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    test_results = {
        "settings": vars(args),
        "model": args.model,
        "primary_aggregation": args.primary_aggregation,
        "n_tasks": args.n_tasks,
        "max_steps": args.max_steps,
        "random_seed": args.random_seed,
        "success_rate": success_rate,
        "tasks_evaluated": len(labels),
        "metrics": metrics,
        "task_details": task_details,
    }

    with open(Path(args.results_dir) / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    logger.info(f"Results saved to {Path(args.results_dir) / 'test_results.json'}")


if __name__ == "__main__":
    main()
