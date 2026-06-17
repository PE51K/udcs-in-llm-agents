#!/usr/bin/env python3
"""Proposed v4 Verification on WebShop Benchmark
================================================

Extends AUQ (Huang et al.) with a separate request uncertainty (u_request) source.
The agent elicits u_request (0.0-1.0) and u_request_explanation per step,
assessed BEFORE choosing an action.

Trajectory-level confidence is aggregated from per-step values using four methods:
    Φ_last    (End-State Belief):    C(τ) = ĉ_T
    Φ_avg     (Overall Quality):     C(τ) = (1/T) Σ ĉ_t
    Φ_min     (Process Reliability): C(τ) = min ĉ_t
    Φ_product (Chain Probability):   C(τ) = Π ĉ_t

For each aggregation three metrics are computed:
    AUROC       - discrimination ability (high confidence → success)
    ECE         - calibration error (T-ECE from paper)
    Brier Score - mean squared error between confidence and outcome

Run from project root:
    export OPENAI_API_KEY=$OPENAI_API_KEY
    uv run --group scripts python -m \
        scripts.verification.webshop.proposed_v4.proposed_v4 \
        --n_tasks 100 --log_level DEBUG --random_seed 42

Arguments:
  --model               LLM model name (default: openrouter/openai/gpt-5.1)
  --primary_aggregation Method highlighted in logs: last|avg|min|product (default: avg)
  --n_tasks             Number of WebShop tasks to run (default: 100)
  --max_steps           Maximum steps per episode (default: 25)
  --success_threshold   Minimum reward to count as success (default: 1.0)
  --results_dir         Output directory (default: ./verification_results/webshop/proposed_v4)
  --log_level           Logging level: DEBUG|INFO|WARNING|ERROR|CRITICAL (default: INFO)
  --random_seed         Random seed for reproducibility (default: None)
  --shuffle_tasks       Randomly shuffle tasks before selection (default: True)
  --include_obs_in_history  Include observation summaries in UAM history (default: True)
  --no-include_obs_in_history  Disable observations in UAM history

Output:
  test_results.json with per-method metrics (auroc/ece/brier) and per-task details
  (step_confidences, c_last, c_avg, c_min, c_product, success, final_reward)
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
import importlib.util
import json
import logging
import os
import random
import re
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

# Add WebShop to sys.path so web_agent_site submodules can be imported
_WEBSHOP_PATH = Path(__file__).parents[4] / "benchmarks" / "WebShop"
if str(_WEBSHOP_PATH) not in sys.path:
    sys.path.insert(0, str(_WEBSHOP_PATH))

# Import WebAgentTextEnv directly from its source file to bypass
# web_agent_site/envs/__init__.py, which eagerly imports WebAgentSiteEnv
# (Selenium-based) and would require selenium as a dependency.
def _import_text_env() -> type:
    # Register a stub package for web_agent_site.envs so Python never
    # executes envs/__init__.py (which imports the Selenium env).
    if "web_agent_site.envs" not in sys.modules:
        stub = types.ModuleType("web_agent_site.envs")
        stub.__path__ = [str(_WEBSHOP_PATH / "web_agent_site" / "envs")]
        stub.__package__ = "web_agent_site.envs"
        sys.modules["web_agent_site.envs"] = stub

    spec = importlib.util.spec_from_file_location(
        "web_agent_site.envs.web_agent_text_env",
        _WEBSHOP_PATH / "web_agent_site" / "envs" / "web_agent_text_env.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["web_agent_site.envs.web_agent_text_env"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod.WebAgentTextEnv

WebAgentTextEnv = _import_text_env()

from openai import OpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_think(text: str) -> str:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_action(text: str) -> str:
    match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(?:\w+)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_confidence(text: str) -> float:
    match = re.search(r"<confidence>\s*([\d.]+)\s*</confidence>", text)
    if not match:
        raise ValueError(
            f"AUQ: missing <confidence> tag in agent output. Output (first 200 chars): {text[:200]}"
        )
    return max(0.0, min(1.0, float(match.group(1))))


def _parse_explanation(text: str) -> str:
    match = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_u_request(text: str) -> float:
    """Parse request uncertainty from <u_request> tags, clamped to [0, 1]."""
    match = re.search(r"<u_request>\s*([\d.]+)\s*</u_request>", text)
    if not match:
        raise ValueError(
            f"Missing <u_request> tag in agent output. Output (first 200 chars): {text[:200]}"
        )
    return max(0.0, min(1.0, float(match.group(1))))


def _parse_u_request_explanation(text: str) -> str:
    match = re.search(r"<u_request_explanation>(.*?)</u_request_explanation>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ProposedV4WebShopAgent:
    """Proposed v4 agent for WebShop: UAM confidence + u_request decomposition.

    Extends AUQ (Huang et al.) with a separate request uncertainty (u_request) source.
    The only difference from AUQ is:
    - Elicits u_request (0.0-1.0) and u_request_explanation per step
    - u_request and u_request_explanation are propagated in UAM history
    """

    def __init__(
        self,
        model_name: str = "openrouter/openai/gpt-5.1",
        temperature: float = 0.0,
        provider: str | None = None,
        include_obs_in_history: bool = True,
    ) -> None:
        self.include_obs_in_history = include_obs_in_history
        self.temperature = temperature

        if model_name.startswith("openrouter/"):
            actual_model_name = model_name.replace("openrouter/", "", 1)
            self.client = OpenAI(
                base_url=os.getenv("BASE_URL"),
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            self.model_name = actual_model_name
        else:
            self.client = OpenAI()
            self.model_name = model_name

        self.provider = provider

        self.uam_history: list[dict] = []
        self.task_agent_token_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.step_index = 0
        self._debug_dump_path: str | None = None
        self._debug_goal_shown: str | None = None

    def _build_system_messages(self) -> str:
        text = (
            "You are a shopping agent. Your goal is to find and buy a product that matches\n"
            "the given instruction on a simulated web store.\n\n"
            "Available actions:\n"
            "  search[keywords]  — search for products using keywords\n"
            "  click[value]      — click a button or link; value must exactly match one\n"
            "                      of the available clickables listed in the observation\n\n"
            "Output format (required):\n"
            "  <think>...</think>\n"
            "  <u_request>0.0-1.0</u_request>\n"
            "  <u_request_explanation>...</u_request_explanation>\n"
            "  <action>search[...] or click[...]</action>\n"
            "  <confidence>0.0-1.0</confidence>\n"
            "  <explanation>...</explanation>"
        )
        return text

    def _build_user_messages(
        self,
        goal: str,
        observation: str,
        available_actions: dict,
        last_action_error: str,
    ) -> str:
        """Build user messages with UAM history and confidence elicitation suffix."""
        parts = []

        # --- Goal ---
        parts.append(f"# Goal\n\n{goal}")

        # --- Current Observation ---
        parts.append(f"# Current Observation\n\n{observation}")

        # --- Available Actions ---
        action_lines = []
        if available_actions.get("has_search_bar"):
            action_lines.append("- search[keywords]")
        clickables = available_actions.get("clickables", [])
        if clickables:
            for c in clickables:
                action_lines.append(f"- click[{c}]")
        action_text = "\n".join(action_lines) if action_lines else "(none)"
        parts.append(f"# Available Actions\n\n{action_text}")

        # --- UAM History (Variant B: Semantic Propagation) ---
        if self.uam_history:
            parts.append("# History of past actions")
            for entry in self.uam_history:
                lines = [f"Step {entry['step']}:"]
                if self.include_obs_in_history and entry.get("observation"):
                    lines.append(f"Observation: {entry['observation']}")
                lines.append(f"<think>{entry['think']}</think>")
                lines.append(f"<u_request>{entry['u_request']}</u_request>")
                lines.append(f"<u_request_explanation>{entry['u_request_explanation']}</u_request_explanation>")
                lines.append(f"<action>{entry['action']}</action>")
                lines.append(f"<confidence>{entry['confidence']}</confidence>")
                lines.append(f"<explanation>{entry['explanation']}</explanation>")
                parts.append("\n".join(lines))

        # --- Last Action Error ---
        if last_action_error:
            parts.append(f"# Error message from last action\n\n{last_action_error}")

        # --- Next Action + u_request assessment + Confidence Elicitation ---
        parts.append(
            "# Next action\n\n"
            f"You are now at step {self.step_index}. "
            f"Prior to this step, you have already taken {self.step_index} step(s).\n\n"
            "Now it's your turn to take an action.\n"
            "You should first reason step-by-step about the current situation. "
            "This reasoning process MUST be enclosed within <think> </think> tags.\n\n"
            "After thinking, you MUST assess your request uncertainty (0.0-1.0) in "
            "<u_request>...</u_request> tags.\n"
            "   0.0 = the goal fully specifies every parameter — there is exactly one correct solution\n"
            "   0.5 = the goal leaves open at least one choice where the user likely has "
            "a specific preference they did not state — you would be guessing on their behalf\n"
            "   1.0 = critical details are missing, many equally valid interpretations exist\n\n"
            "Be meticulous: if the goal leaves ANY parameter open-ended, ask yourself — "
            "would a real user genuinely be satisfied with ANY valid option, "
            "or do they most likely have a specific intent they failed to communicate? "
            "If you find yourself choosing one option among several equally plausible "
            "alternatives without a clear basis, that is a sign u_request should be high.\n\n"
            "Then explain your assessment in <u_request_explanation>...</u_request_explanation> tags.\n\n"
            "Once you've finished your reasoning, you should choose an action "
            "and present it within <action> </action> tags.\n\n"
            "After your action, you MUST provide:\n\n"
            "1. Your confidence level (0.0-1.0) in <confidence>...</confidence> tags\n\n"
            "2. An explanation of your confidence in <explanation>...</explanation> tags\n"
            "   - Explain what makes you confident\n"
            "   - Explain what concerns or uncertainties you have\n"
            "   - What information might be missing or unclear\n"
            "   - What alternative actions you considered\n"
            "   - DO NOT output empty <explanation></explanation> tags - "
            "you MUST provide actual text inside"
        )

        return "\n".join(parts)

    def _query_model(
        self, system_msg: str, user_msg: str
    ) -> tuple[str, dict]:
        params: dict = {}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.provider:
            # --provider may be a single slug or a comma-separated ordered list.
            _order = self.provider if isinstance(self.provider, list) else [
                p.strip() for p in str(self.provider).split(",") if p.strip()
            ]
            params["extra_body"] = {
                "provider": {"order": _order, "allow_fallbacks": False}
            }
        # Optionally disable the provider reasoning channel (set per-backbone
        # by run_parallel.sh) so the model emits the full tagged output inline
        # in `content` instead of omitting pre-action tags like <u_request>.
        if os.getenv("OPENROUTER_DISABLE_REASONING") == "1":
            params.setdefault("extra_body", {})["reasoning"] = {"enabled": False}

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # --- Debug: dump full LLM input/output to file ---
        if self._debug_dump_path:
            with open(self._debug_dump_path, "a") as df:
                df.write(f"\n{'='*80}\n")
                df.write(f"STEP {self.step_index}  (goal_shown: {getattr(self, '_debug_goal_shown', 'N/A')})\n")
                df.write(f"{'='*80}\n")
                for msg in messages:
                    df.write(f"\n--- role: {msg['role']} ---\n")
                    content = msg["content"]
                    if isinstance(content, list):
                        for part in content:
                            df.write(part.get("text", str(part)) + "\n")
                    else:
                        df.write(str(content) + "\n")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            n=1,
            **params,
        )

        # OpenRouter routes some providers' tagged output into `reasoning`;
        # reconstruct the full text before parsing (content may be None/empty).
        _msg = response.choices[0].message
        _reasoning = getattr(_msg, "reasoning", None) or getattr(_msg, "reasoning_content", None) or ""
        result_text = (_reasoning + "\n" + (_msg.content or "")).strip() if _reasoning else (_msg.content or "")

        if self._debug_dump_path:
            with open(self._debug_dump_path, "a") as df:
                df.write(f"\n--- LLM OUTPUT ---\n")
                df.write(result_text + "\n")

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            logger.debug(
                f"  API usage: prompt={usage['prompt_tokens']}, "
                f"completion={usage['completion_tokens']}, total={usage['total_tokens']}"
            )
        return result_text, usage

    def get_action(
        self,
        goal: str,
        observation: str,
        available_actions: dict,
        last_action_error: str = "",
    ) -> tuple[str, dict]:
        """Get action and update UAM history.

        Returns:
            action: Parsed action string (e.g. "search[laptop]" or "click[Buy Now]")
            agent_info: Dict with step_confidence, step_u_request, step_explanation, token usage
        """
        logger.info(f"Getting action for step {self.step_index}")

        system_msgs = self._build_system_messages()
        user_msgs = self._build_user_messages(
            goal, observation, available_actions, last_action_error
        )
        raw_text, step_usage = self._query_model(system_msgs, user_msgs)
        logger.debug(f"  Full model output:\n{raw_text}")

        # Accumulate task-level token usage
        for k in self.task_agent_token_usage:
            self.task_agent_token_usage[k] += step_usage.get(k, 0)

        think = _parse_think(raw_text)
        u_request = _parse_u_request(raw_text)
        u_request_explanation = _parse_u_request_explanation(raw_text)
        action = _parse_action(raw_text)
        confidence = _parse_confidence(raw_text)
        explanation = _parse_explanation(raw_text)

        logger.info(
            f"Step {self.step_index}: action={action!r}, confidence={confidence:.4f}, u_request={u_request:.4f}"
        )
        logger.debug(f"  think={think[:80]}...")
        logger.debug(f"  explanation={explanation[:80]}...")
        logger.debug(f"  u_request_explanation={u_request_explanation[:80]}...")

        self.uam_history.append(
            {
                "step": self.step_index,
                "observation": observation,
                "think": think,
                "u_request": u_request,
                "u_request_explanation": u_request_explanation,
                "action": action,
                "confidence": confidence,
                "explanation": explanation,
            }
        )
        self.step_index += 1

        return action, {
            "step_confidence": confidence,
            "step_u_request": u_request,
            "step_explanation": explanation,
            "step_think": think,
            "step_action": action,
            "step_u_request_explanation": u_request_explanation,
            "step_agent_token_usage": step_usage,
            "task_agent_token_usage": dict(self.task_agent_token_usage),
        }


# ---------------------------------------------------------------------------
# TestRunner
# ---------------------------------------------------------------------------


class TestRunner:
    """Runs Proposed v4 agent on WebShop benchmark sessions."""

    def __init__(
        self,
        model_name: str = "openrouter/openai/gpt-5.1",
        provider: str | None = None,
        include_obs_in_history: bool = True,
        results_dir: str = "./verification_results/webshop/proposed_v4",
        n_tasks: int = 100,
        max_steps: int = 25,
        success_threshold: float = 1.0,
        shuffle_tasks: bool = True,
        random_seed: int | None = None,
    ):
        self.model_name = model_name
        self.provider = provider
        self.include_obs_in_history = include_obs_in_history
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_dir = self.results_dir / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.n_tasks = n_tasks
        self.max_steps = max_steps
        self.success_threshold = success_threshold
        self.shuffle_tasks = shuffle_tasks
        self.random_seed = random_seed

    def run_agent(self) -> tuple[list[list[float]], list[list[float]], list[int], dict[str, Any]]:
        """Run agent on WebShop sessions.

        Returns:
            all_step_confidences: Per-step confidence lists for each episode
            all_step_u_requests: Per-step u_request lists for each episode
            labels: Binary success labels (1=success, 0=failure)
            task_details: Per-session detail dict
        """
        logger.info(f"Running proposed v4 agent on WebShop: model={self.model_name}")

        env = WebAgentTextEnv(observation_mode="text", num_products=None)

        # Build session list from all available goals in the environment
        all_sessions = list(range(len(env.server.goals)))
        if self.shuffle_tasks:
            random.shuffle(all_sessions)
            logger.info(f"Sessions shuffled, selecting first {self.n_tasks}")
        task_sessions = all_sessions[: self.n_tasks]

        all_step_confidences: list[list[float]] = []
        all_step_u_requests: list[list[float]] = []
        labels: list[int] = []
        task_details: dict[str, Any] = {}

        # Load already-completed task results
        for task_file in sorted(self.tasks_dir.glob("task_*.json")):
            with open(task_file) as f:
                td = json.load(f)
            task_key = str(td["session_idx"])
            task_details[task_key] = td
            all_step_confidences.append(td["step_confidences"])
            all_step_u_requests.append(td.get("step_u_requests", [0.0] * len(td["step_confidences"])))
            labels.append(td["success"])
            logger.info(f"Loaded existing result for session {td['session_idx']}")

        completed_sessions = {int(k) for k in task_details}

        sessions_to_run = [s for s in task_sessions if s not in completed_sessions]
        logger.info(
            f"Sessions to run: {len(sessions_to_run)} "
            f"(skipping {len(completed_sessions)} already completed)"
        )

        for session_idx in sessions_to_run:
            logger.info(f"--- Session {session_idx} ---")
            try:
                obs, _ = env.reset(session=session_idx)
                goal = env.instruction_text
                logger.info(f"  Goal: {goal[:100]}...")

                agent = ProposedV4WebShopAgent(
                    model_name=self.model_name, provider=self.provider,
                    include_obs_in_history=self.include_obs_in_history,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    debug_dir = self.results_dir / "debug_dumps"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    dump_path = debug_dir / f"session_{session_idx}.txt"
                    agent._debug_dump_path = str(dump_path)
                    agent._debug_goal_shown = goal
                    with open(dump_path, "w") as df:
                        df.write(f"Session: {session_idx}\n")
                        df.write(f"Goal: {goal}\n")
                        df.write(f"Initial observation: {obs}\n")

                step_confidences: list[float] = []
                step_u_requests: list[float] = []
                step_traces: list[dict] = []
                final_reward = 0.0

                for step in range(self.max_steps):
                    available = env.get_available_actions()
                    action, info = agent.get_action(
                        goal=goal,
                        observation=obs,
                        available_actions=available,
                        last_action_error="",
                    )
                    step_confidences.append(info["step_confidence"])
                    step_u_requests.append(info["step_u_request"])
                    step_traces.append({
                        "step": step,
                        "think": info["step_think"],
                        "action": info["step_action"],
                        "confidence": info["step_confidence"],
                        "explanation": info["step_explanation"],
                        "u_request": info["step_u_request"],
                        "u_request_explanation": info["step_u_request_explanation"],
                    })

                    obs, reward, done, _ = env.step(action)
                    final_reward = reward

                    logger.debug(
                        f"  step={step}, action={action!r}, "
                        f"reward={reward}, done={done}"
                    )

                    if done:
                        break

                success = 1 if final_reward >= self.success_threshold else 0
                all_step_confidences.append(step_confidences)
                all_step_u_requests.append(step_u_requests)
                labels.append(success)

                task_key = str(session_idx)
                task_details[task_key] = {
                    "session_idx": session_idx,
                    "goal": goal,
                    "step_confidences": step_confidences,
                    "step_u_requests": step_u_requests,
                    "step_traces": step_traces,
                    "success": success,
                    "final_reward": float(final_reward),
                    "task_agent_token_usage": info.get("task_agent_token_usage", {}),
                    "n_steps": len(step_confidences),
                }
                task_file = self.tasks_dir / f"task_{session_idx}.json"
                with open(task_file, "w") as f:
                    json.dump(task_details[task_key], f, indent=2)
                logger.debug(f"  Saved result for session {session_idx} to {task_file}")

                logger.info(
                    f"  Session {session_idx}: steps={len(step_confidences)}, "
                    f"success={success}, reward={final_reward:.4f}, "
                    f"tokens={info.get('task_agent_token_usage', {}).get('total_tokens', '?')}"
                )

            except Exception as e:
                logger.error(f"Error on session {session_idx}: {e}", exc_info=True)

        env.close()
        logger.info(f"Completed {len(labels)} episodes")
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
        u_request aggregations: first, max, avg (inverted for ROC-AUC: high u_request = low confidence)

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Proposed v4 verification on WebShop")
    parser.add_argument(
        "--model", default=os.getenv("MODEL", "openrouter/openai/gpt-5.1"), help="LLM model name"
    )
    parser.add_argument(
        "--primary_aggregation",
        default="avg",
        choices=["last", "avg", "min", "product"],
        help="Primary aggregation method highlighted in logs",
    )
    parser.add_argument(
        "--n_tasks", type=int, default=100, help="Number of WebShop sessions to run"
    )
    parser.add_argument(
        "--max_steps", type=int, default=25, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--success_threshold",
        type=float,
        default=1.0,
        help="Minimum reward to count as success (default: 1.0)",
    )
    parser.add_argument(
        "--results_dir",
        default="./verification_results/webshop/proposed_v4",
        help="Output directory",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument(
        "--shuffle_tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly shuffle sessions before selection",
    )
    parser.add_argument(
        "--include_obs_in_history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include observation summaries in UAM history (paper Variant B)",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("PROVIDER"),
        help="OpenRouter provider slug(s) to pin, comma-separated for an ordered allowlist (e.g. DeepInfra,Parasail); disables fallback. Default none (auto-route).",
    )
    args = parser.parse_args()

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

    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        logger.info(f"Random seed set to: {args.random_seed}")

    logger.info("=" * 60)
    logger.info("Proposed v4 Verification on WebShop")
    logger.info(
        f"Model: {args.model}, Tasks: {args.n_tasks}, "
        f"obs_in_history: {args.include_obs_in_history}"
    )
    logger.info("=" * 60)

    runner = TestRunner(
        model_name=args.model, provider=args.provider,
        include_obs_in_history=args.include_obs_in_history,
        results_dir=args.results_dir,
        n_tasks=args.n_tasks,
        max_steps=args.max_steps,
        success_threshold=args.success_threshold,
        shuffle_tasks=args.shuffle_tasks,
        random_seed=args.random_seed,
    )

    all_step_confidences, all_step_u_requests, labels, task_details = runner.run_agent()

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
        "success_threshold": args.success_threshold,
        "random_seed": args.random_seed,
        "success_rate": success_rate,
        "tasks_evaluated": len(labels),
        "metrics": metrics,
        "task_details": task_details,
    }

    output_path = Path(args.results_dir) / "test_results.json"
    with open(output_path, "w") as f:
        json.dump(test_results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
