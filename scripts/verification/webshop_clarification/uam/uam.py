#!/usr/bin/env python3
"""AUQ Clarification Benchmark on WebShop
================================================

Clarification-seeking variant of the AUQ baseline. 50% of tasks are
underspecified (attributes/options stripped from instruction). The agent
has a request_clarification action to request a more specified goal. If the
agent requests clarification on an underspecified task, the full instruction is
revealed and the episode continues. Evaluates clarification precision/recall/F1/accuracy
alongside standard fault-detection metrics.

Run from project root:
    export OPENAI_API_KEY=$OPENAI_API_KEY
    uv run --group scripts python -m \
        scripts.verification.webshop_clarification.uam.uam \
        --n_tasks 100 --log_level INFO --random_seed 42
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
# Parsing helpers (reused from hackable.py)
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


# ---------------------------------------------------------------------------
# Clarification helpers
# ---------------------------------------------------------------------------

def _make_underspecified_webshop(instruction: str, goal: dict) -> str:
    """Build an underspecified version by removing attributes and option clauses.

    Strips attribute words and entire ``key: value`` option clauses from the
    instruction, keeping only the product type and price constraint so the
    result reads naturally without obvious blanks.
    """
    goal_options = goal.get('goal_options', {})

    # 1. Extract the price clause (always at the end)
    price_match = re.search(
        r',?\s*and price lower than [\d.]+ dollars\s*$', instruction, re.IGNORECASE
    )
    price_clause = price_match.group(0) if price_match else ''
    base = instruction[:price_match.start()] if price_match else instruction

    # 2. Remove entire option clauses ("key: value" pairs)
    if isinstance(goal_options, dict) and goal_options:
        # First pass: ", and key: value" (non-first options)
        for opt_key, opt_val in goal_options.items():
            pat = (r',\s*and\s+' + re.escape(str(opt_key))
                   + r':\s*' + re.escape(str(opt_val)))
            base = re.sub(pat, '', base, flags=re.IGNORECASE)
        # Second pass: " with key: value" (first option in the section)
        for opt_key, opt_val in goal_options.items():
            pat = (r'\s+with\s+' + re.escape(str(opt_key))
                   + r':\s*' + re.escape(str(opt_val)))
            base = re.sub(pat, '', base, flags=re.IGNORECASE)

    # 3. Strip attribute words from the base instruction
    for attr in goal.get('attributes', []):
        base = re.sub(r'\b' + re.escape(attr) + r'\b', '', base, flags=re.IGNORECASE)

    # 4. Clean up residual punctuation and connectors
    base = re.sub(r',(\s*,)+', ',', base)          # consecutive commas
    base = re.sub(r'\bwith\s*,', 'with', base)     # "with ,"
    base = re.sub(r'\bwith\s+for\b', 'for', base)  # "with for"
    base = re.sub(r'\bwith\s+and\b', '', base)      # "with and"
    base = re.sub(r'\bfor\s*,', 'for', base)        # "for ,"
    base = re.sub(r'\bwith\s*$', '', base)           # trailing "with"
    base = re.sub(r'\bfor\s*$', '', base)            # trailing "for"
    base = re.sub(r',\s*$', '', base)                # trailing comma
    base = re.sub(r'(Find me|I need|I want)\s*,\s*', r'\1 ', base, flags=re.IGNORECASE)
    base = re.sub(r'\s+', ' ', base).strip()

    # 5. Reassemble with price clause
    result = base + price_clause
    return re.sub(r'\s+', ' ', result).strip()


def _build_clarification_task_set(n_tasks: int, random_seed: int | None = None) -> list[int]:
    """Generate labels: 0=specified, 1=underspecified. 50/50 split."""
    rng = random.Random(random_seed)
    labels = [0] * (n_tasks // 2) + [1] * (n_tasks - n_tasks // 2)
    rng.shuffle(labels)
    return labels


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AUQWebShopAgent:
    """Standalone AUQ agent for WebShop text environment.

    Implements System 1 from the AUQ paper (Huang et al.):
    - Confidence Elicitation Protocol: structured <think>/<action>/<confidence>/<explanation>
    - Uncertainty-Aware Memory (UAM): past actions, confidence scores, and explanations
      are propagated into the prompt (Variant B: Semantic Propagation)
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
            "  search[keywords]      — search for products using keywords\n"
            "  click[value]          — click a button or link; value must exactly match one\n"
            "                          of the available clickables listed in the observation\n"
            "  request_clarification  — request a more specified goal if the request is\n"
            "                          missing key details or has multiple valid solutions (e.g. color, size)\n\n"
            "Output format (required):\n"
            "  <think>...</think>\n"
            "  <action>search[...] or click[...] or request_clarification</action>\n"
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
        """Build user messages with UAM history and confidence elicitation suffix.

        Structure:
        1. Goal (shopping instruction)
        2. Current observation (text from env)
        3. Available actions
        4. UAM History (paper Variant B: each step has obs?, think, action, confidence, explanation)
        5. Last action error (if any)
        6. Next action + confidence elicitation suffix
        """
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
        action_lines.append(
            "- request_clarification — request a more specified goal if the "
            "request is missing key details or has multiple valid solutions (e.g. color, size)"
        )
        action_text = "\n".join(action_lines) if action_lines else "(none)"
        parts.append(f"# Available Actions\n\n{action_text}")

        # --- UAM History (paper Variant B: Semantic Propagation) ---
        if self.uam_history:
            parts.append("# History of past actions")
            for entry in self.uam_history:
                lines = [f"Step {entry['step']}:"]
                if self.include_obs_in_history and entry.get("observation"):
                    lines.append(f"Observation: {entry['observation']}")
                lines.append(
                    f"Action: <think>{entry['think']}</think> "
                    f"<action>{entry['action']}</action>"
                )
                lines.append(f"<confidence>{entry['confidence']}</confidence>")
                lines.append(f"<explanation>{entry['explanation']}</explanation>")
                parts.append("\n".join(lines))

        # --- Last Action Error ---
        if last_action_error:
            parts.append(f"# Error message from last action\n\n{last_action_error}")

        # --- Next Action + Confidence Elicitation ---
        parts.append(
            "# Next action\n\n"
            f"You are now at step {self.step_index}. "
            f"Prior to this step, you have already taken {self.step_index} step(s).\n\n"
            "Now it's your turn to take an action.\n\n"
            "If the goal is ambiguous or missing key details, "
            "you should seek clarification before acting.\n\n"
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

        if self._debug_dump_path:
            with open(self._debug_dump_path, "a") as df:
                df.write(f"\n{'='*80}\n")
                df.write(f"STEP {self.step_index}  (goal_shown: {self._debug_goal_shown})\n")
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
            agent_info: Dict with step_confidence, step_explanation, token usage
        """
        logger.info(f"Getting action for step {self.step_index}")

        system_msgs = self._build_system_messages()
        user_msgs = self._build_user_messages(
            goal, observation, available_actions, last_action_error
        )
        raw_text, step_usage = self._query_model(system_msgs, user_msgs)

        # Accumulate task-level token usage
        for k in self.task_agent_token_usage:
            self.task_agent_token_usage[k] += step_usage.get(k, 0)

        think = _parse_think(raw_text)
        action = _parse_action(raw_text)
        confidence = _parse_confidence(raw_text)
        explanation = _parse_explanation(raw_text)

        logger.info(
            f"Step {self.step_index}: action={action!r}, confidence={confidence:.4f}"
        )
        logger.debug(f"  think={think[:80]}...")
        logger.debug(f"  explanation={explanation[:80]}...")

        self.uam_history.append(
            {
                "step": self.step_index,
                "observation": observation,
                "think": think,
                "action": action,
                "confidence": confidence,
                "explanation": explanation,
            }
        )
        self.step_index += 1

        return action, {
            "step_confidence": confidence,
            "step_explanation": explanation,
            "step_think": think,
            "step_action": action,
            "step_agent_token_usage": step_usage,
            "task_agent_token_usage": dict(self.task_agent_token_usage),
        }


# ---------------------------------------------------------------------------
# TestRunner
# ---------------------------------------------------------------------------


class TestRunner:
    """Runs AUQ agent on WebShop clarification benchmark."""

    def __init__(
        self,
        model_name: str = "openrouter/openai/gpt-5.1",
        provider: str | None = None,
        include_obs_in_history: bool = True,
        results_dir: str = "./verification_results/webshop_clarification/uam",
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

    def run_agent(self) -> tuple[list[list[float]], list[int], list[int], list[int], dict[str, Any]]:
        """Run agent on WebShop clarification benchmark.

        Returns:
            all_step_confidences, labels, task_labels, all_agent_asked, task_details
        """
        logger.info(f"Running AUQ clarification agent on WebShop: model={self.model_name}")

        env = WebAgentTextEnv(observation_mode="text", num_products=None)

        all_sessions = list(range(len(env.server.goals)))
        if self.shuffle_tasks:
            random.shuffle(all_sessions)
        task_sessions = all_sessions[: self.n_tasks]

        clarification_labels = _build_clarification_task_set(self.n_tasks, self.random_seed)

        all_step_confidences: list[list[float]] = []
        labels: list[int] = []
        task_labels: list[int] = []
        all_agent_asked: list[int] = []
        task_details: dict[str, Any] = {}

        # Load already-completed task results
        for task_file in sorted(self.tasks_dir.glob("task_*.json")):
            with open(task_file) as f:
                td = json.load(f)
            task_key = str(td["session_idx"])
            task_details[task_key] = td
            all_step_confidences.append(td["step_confidences"])
            labels.append(td["success"])
            task_labels.append(td["task_label"])
            all_agent_asked.append(td["agent_asked"])
            logger.info(f"Loaded existing result for session {td['session_idx']}")

        completed_sessions = {int(k) for k in task_details}

        for task_idx, session_idx in enumerate(task_sessions):
            if session_idx in completed_sessions:
                continue

            task_label = clarification_labels[task_idx]
            logger.info(
                f"--- Session {session_idx} (task {task_idx}, "
                f"{'underspecified' if task_label else 'specified'}) ---"
            )
            try:
                obs, _ = env.reset(session=session_idx)
                original_goal = env.instruction_text

                if task_label == 1:
                    goal_data = env.server.goals[session_idx]
                    goal = _make_underspecified_webshop(original_goal, goal_data)
                    goal_shown = goal  # save before potential reveal
                    # Strip original goal from observation so it doesn't leak.
                    # The WebShop observation embeds the instruction as bare text
                    # (without the "Instruction: " prefix), so we must match on
                    # the bare goal text rather than the full instruction_text string.
                    bare_original = original_goal.removeprefix("Instruction: ")
                    bare_goal = goal.removeprefix("Instruction: ")
                    obs = obs.replace(bare_original, bare_goal, 1)
                    if bare_original in obs:
                        raise RuntimeError(f"LEAK: original goal still in initial obs after sanitization (session {session_idx})")
                    else:
                        logger.debug(f"  Sanitized initial obs: replaced original goal")
                    logger.info(f"  Original: {original_goal[:100]}...")
                    logger.info(f"  Underspecified: {goal[:100]}...")
                else:
                    goal = original_goal
                    goal_shown = goal
                    bare_original = bare_goal = ""
                    logger.info(f"  Goal: {goal[:100]}...")

                agent = AUQWebShopAgent(
                    model_name=self.model_name, provider=self.provider,
                    include_obs_in_history=self.include_obs_in_history,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    debug_dir = self.results_dir / "debug_dumps"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    dump_path = debug_dir / f"session_{session_idx}_label{task_label}.txt"
                    agent._debug_dump_path = str(dump_path)
                    agent._debug_goal_shown = goal_shown if task_label == 1 else goal
                    with open(dump_path, "w") as df:
                        df.write(f"Session: {session_idx}\n")
                        df.write(f"Task label: {task_label} ({'underspecified' if task_label else 'specified'})\n")
                        df.write(f"Original goal: {original_goal}\n")
                        df.write(f"Goal shown to agent: {goal_shown if task_label == 1 else goal}\n")
                        df.write(f"Initial observation (first 500 chars): {obs[:500]}\n")

                step_confidences: list[float] = []
                step_traces: list[dict] = []
                final_reward = 0.0
                agent_asked = 0

                for step in range(self.max_steps):
                    available = env.get_available_actions()
                    action, info = agent.get_action(
                        goal=goal,
                        observation=obs,
                        available_actions=available,
                        last_action_error="",
                    )
                    step_confidences.append(info["step_confidence"])
                    step_traces.append({
                        "step": step,
                        "think": info["step_think"],
                        "action": info["step_action"],
                        "confidence": info["step_confidence"],
                        "explanation": info["step_explanation"],
                    })

                    if action.lower().startswith("request_clarification"):
                        agent_asked = 1
                        logger.info(f"  Step {step}: agent requested clarification")
                        if task_label == 1:
                            goal = original_goal
                            logger.info(f"  Revealed full goal: {original_goal[:100]}...")
                        continue

                    obs, reward, done, _ = env.step(action)
                    # Sanitize observation while goal is still underspecified
                    if task_label == 1 and not agent_asked:
                        obs = obs.replace(bare_original, bare_goal, 1)
                        if bare_original in obs:
                            raise RuntimeError(f"LEAK: original goal still in step {step} obs after sanitization (session {session_idx})")
                    final_reward = reward

                    logger.debug(
                        f"  step={step}, action={action!r}, "
                        f"reward={reward}, done={done}"
                    )

                    if done:
                        break

                success = 1 if final_reward >= self.success_threshold else 0
                all_step_confidences.append(step_confidences)
                labels.append(success)
                task_labels.append(task_label)
                all_agent_asked.append(agent_asked)

                task_key = str(session_idx)
                task_details[task_key] = {
                    "session_idx": session_idx,
                    "goal": original_goal,
                    "goal_shown": goal_shown,
                    "task_label": task_label,
                    "agent_asked": agent_asked,
                    "step_confidences": step_confidences,
                    "step_traces": step_traces,
                    "success": success,
                    "final_reward": float(final_reward),
                    "task_agent_token_usage": info.get("task_agent_token_usage", {}),
                    "n_steps": len(step_confidences),
                }
                task_file = self.tasks_dir / f"task_{session_idx}.json"
                with open(task_file, "w") as f:
                    json.dump(task_details[task_key], f, indent=2)

                logger.info(
                    f"  Session {session_idx}: steps={len(step_confidences)}, "
                    f"success={success}, asked={agent_asked}, reward={final_reward:.4f}"
                )

            except Exception as e:
                logger.error(f"Error on session {session_idx}: {e}", exc_info=True)

        env.close()
        logger.info(f"Completed {len(labels)} episodes")
        return all_step_confidences, labels, task_labels, all_agent_asked, task_details

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
        all_step_confidences: list[list[float]], labels: list[int]
    ) -> dict[str, dict[str, float]]:
        """Compute AUROC, ECE, and Brier Score for confidence aggregations."""
        fallback = {"auroc": 0.5, "ece": 1.0, "brier": 1.0}
        if len(set(labels)) < 2:
            logger.warning("Only one class present, returning fallback metrics")
            return {f"confidence/{m}": fallback.copy() for m in ("last", "avg", "min", "product")}

        results: dict[str, dict[str, float]] = {}
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

        return results

    @staticmethod
    def compute_clarification_metrics(
        all_agent_asked: list[int], task_labels: list[int]
    ) -> dict[str, float]:
        tp = sum(1 for a, l in zip(all_agent_asked, task_labels) if a == 1 and l == 1)
        fp = sum(1 for a, l in zip(all_agent_asked, task_labels) if a == 1 and l == 0)
        fn = sum(1 for a, l in zip(all_agent_asked, task_labels) if a == 0 and l == 1)
        tn = sum(1 for a, l in zip(all_agent_asked, task_labels) if a == 0 and l == 0)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        acc = (tp + tn) / len(task_labels) if len(task_labels) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f1, "accuracy": acc}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="AUQ clarification benchmark on WebShop")
    parser.add_argument(
        "--model", default=os.getenv("MODEL", "openrouter/openai/gpt-5.1"), help="LLM model name"
    )
    parser.add_argument(
        "--primary_aggregation",
        default="avg",
        choices=["last", "avg", "min", "product"],
    )
    parser.add_argument(
        "--n_tasks", type=int, default=100, help="Number of WebShop sessions to run"
    )
    parser.add_argument(
        "--max_steps", type=int, default=25, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--success_threshold", type=float, default=1.0,
    )
    parser.add_argument(
        "--results_dir",
        default="./verification_results/webshop_clarification/uam",
    )
    parser.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument(
        "--shuffle_tasks", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--include_obs_in_history", action=argparse.BooleanOptionalAction, default=True,
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

    logger.info("=" * 60)
    logger.info("AUQ Clarification Benchmark on WebShop")
    logger.info(f"Model: {args.model}, Tasks: {args.n_tasks}")
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

    all_step_confidences, labels, task_labels, all_agent_asked, task_details = runner.run_agent()

    if not labels:
        logger.error("No results, cannot compute metrics")
        return

    metrics = runner.compute_metrics(all_step_confidences, labels)
    clarification_metrics = runner.compute_clarification_metrics(all_agent_asked, task_labels)
    success_rate = float(np.mean(labels))
    n_specified = sum(1 for l in task_labels if l == 0)
    n_underspecified = sum(1 for l in task_labels if l == 1)

    logger.info("=" * 60)
    logger.info("Test Results:")
    for method, m in metrics.items():
        logger.info(f"  {method}: AUROC={m['auroc']:.4f}  ECE={m['ece']:.4f}  Brier={m['brier']:.4f}")
    logger.info(f"  Success Rate: {success_rate:.2%}")
    logger.info(f"  Tasks: {len(labels)} ({n_specified} specified, {n_underspecified} underspecified)")
    logger.info(
        f"  Clarification: P={clarification_metrics['precision']:.4f} "
        f"R={clarification_metrics['recall']:.4f} "
        f"F1={clarification_metrics['f1']:.4f} "
        f"Acc={clarification_metrics['accuracy']:.4f}"
    )
    logger.info("=" * 60)

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
        "n_specified": n_specified,
        "n_underspecified": n_underspecified,
        "metrics": metrics,
        "clarification_metrics": clarification_metrics,
        "task_details": task_details,
    }

    output_path = Path(args.results_dir) / "test_results.json"
    with open(output_path, "w") as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
