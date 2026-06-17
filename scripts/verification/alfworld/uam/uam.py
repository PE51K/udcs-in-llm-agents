#!/usr/bin/env python3
"""AUQ ROC-AUC Verification on ALfWorld Benchmark
==================================================

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
    export OPENAI_API_KEY=$OPENAI_API_KEY
    export ALFWORLD_DATA=<path>
    uv run --group scripts python -m \
        scripts.verification.alfworld.uam.uam \
        --n_tasks 100 --log_level DEBUG --random_seed 42

Arguments:
  --model               LLM model name (default: openrouter/openai/gpt-5.1)
  --primary_aggregation Method highlighted in logs: last|avg|min|product (default: avg)
  --n_tasks             Number of ALfWorld tasks to run (default: 100)
  --max_steps           Maximum steps per episode (default: 25)
  --config_path         Path to ALfWorld config YAML (default: benchmarks/alfworld/configs/base_config.yaml)
  --task_types          Task type IDs to include: 1-6 (default: all)
  --results_dir         Output directory (default: ./verification_results/alfworld/uam)
  --log_level           Logging level: DEBUG|INFO|WARNING|ERROR|CRITICAL (default: INFO)
  --random_seed         Random seed for reproducibility (default: None)
  --shuffle_tasks       Randomly shuffle tasks before selection (default: True)
  --include_obs_in_history  Include observation summaries in UAM history (default: True)
  --no-include_obs_in_history  Disable observations in UAM history

Output:
  test_results.json with per-method metrics (auroc/ece/brier) and per-task details
  (step_confidences, c_last, c_avg, c_min, c_product, success, task_type)
"""

from dotenv import load_dotenv

load_dotenv()

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import brier_score_loss, roc_auc_score

# Add ALfWorld to sys.path so alfworld submodules can be imported
_ALFWORLD_PATH = Path(__file__).parents[4] / "benchmarks" / "alfworld"
if str(_ALFWORLD_PATH) not in sys.path:
    sys.path.insert(0, str(_ALFWORLD_PATH))

from alfworld.agents.environment import get_environment

from openai import OpenAI

logger = logging.getLogger(__name__)

# ALfWorld task type mapping (from alfred_tw_env.py)
TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}

TASK_TYPE_LABELS = {
    1: "Pick & Place",
    2: "Examine in Light",
    3: "Clean & Place",
    4: "Heat & Place",
    5: "Cool & Place",
    6: "Pick Two & Place",
}


# ---------------------------------------------------------------------------
# Config & parsing helpers
# ---------------------------------------------------------------------------


def _load_alfworld_config(config_path: str) -> dict:
    """Load ALfWorld YAML config directly (avoids argparse conflicts with
    alfworld.agents.modules.generic.load_config)."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def _extract_goal(observation: str) -> str:
    """Parse 'Your task is to: ...' from the initial ALfWorld observation."""
    match = re.search(r"Your task is to:\s*(.+)", observation)
    if match:
        return match.group(1).strip()
    # Fallback: return entire observation if pattern not found
    return observation.strip()


def _get_task_type_from_path(game_file_path: str) -> str | None:
    """Extract task type name from a game file path.

    Game file paths contain the task type directory, e.g.:
    .../pick_and_place_simple-Pen-1/trial_T.../game.tw-pddl
    """
    for task_type_name in TASK_TYPES.values():
        if task_type_name in game_file_path:
            return task_type_name
    return None


def _get_task_type_id(task_type_name: str) -> int | None:
    """Get numeric task type ID from name."""
    for tid, tname in TASK_TYPES.items():
        if tname == task_type_name:
            return tid
    return None


def _validate_action(agent_action: str, admissible_commands: list[str]) -> str:
    """Fuzzy-match agent output to admissible commands.

    Returns the best matching admissible command, or the original action
    if no match is found (the env will handle invalid actions).
    """
    # Exact match
    if agent_action in admissible_commands:
        return agent_action

    # Case-insensitive match
    lower_action = agent_action.lower().strip()
    for cmd in admissible_commands:
        if cmd.lower().strip() == lower_action:
            return cmd

    # Substring match (agent output contains an admissible command)
    for cmd in admissible_commands:
        if cmd.lower() in lower_action:
            return cmd

    # No match found — return original and let env handle it
    logger.warning(
        f"Action {agent_action!r} not in admissible commands, "
        f"passing through. Admissible: {admissible_commands[:5]}..."
    )
    return agent_action


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
# Agent
# ---------------------------------------------------------------------------


class AUQAlfWorldAgent:
    """Standalone AUQ agent for ALfWorld text environment.

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

    def _build_system_messages(self) -> list[dict]:
        """Build system message following the paper's baseline prompt structure.

        Paper Appendix A.4 — Baseline System Prompt for ALFWorld:
        The system message establishes the agent's role. Task-specific details
        (goal, observation, history, admissible actions) go in user messages.
        """
        text = (
            "You are an expert agent operating in the ALFRED Embodied Environment."
        )
        return [{"type": "text", "text": text}]

    def _build_user_messages(
        self,
        goal: str,
        observation: str,
        admissible_commands: list[str],
        last_action_error: str,
    ) -> list[dict]:
        """Build user messages following the paper's prompt template (Appendix A.4).

        Paper structure (single integrated prompt):
        - Task description (goal)
        - Step count + UAM history (Variant B: obs, think, action, confidence, explanation)
        - Current step + observation + admissible actions
        - ReAct instruction (<think> then <action>)
        - Confidence Elicitation Suffix
        """
        parts = []

        # --- Task description ---
        parts.append(f"Your task is to: {goal}")

        # --- Step count + UAM History ---
        history_len = len(self.uam_history)
        parts.append(
            f"Prior to this step, you have already taken {self.step_index} step(s)."
        )
        if self.uam_history:
            # Paper Variant B: Semantic Propagation (Confidence + Explanation)
            parts.append(
                f"Below are the most recent {history_len} observations and "
                "the corresponding actions you took:"
            )
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
            parts.append(f"Error from last action: {last_action_error}")

        # --- Current step + observation + admissible actions ---
        admissible_str = ", ".join(admissible_commands) if admissible_commands else "(none)"
        parts.append(
            f"You are now at step {self.step_index} and your current observation is: "
            f"{observation}\n"
            f"Your admissible actions of the current situation are: [{admissible_str}]."
        )

        # --- ReAct instruction (paper baseline prompt) ---
        parts.append(
            "Now it's your turn to take an action.\n"
            "You should first reason step-by-step about the current situation. "
            "This reasoning process MUST be enclosed within <think> </think> tags.\n"
            "Once you've finished your reasoning, you should choose an admissible "
            "action for the current step and present it within <action> </action> tags."
        )

        # --- Confidence Elicitation Suffix (paper Appendix A.4) ---
        parts.append(
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

        return [{"type": "text", "text": "\n\n".join(parts)}]

    def _query_model(
        self, system_msgs: list[dict], user_msgs: list[dict]
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
            {"role": "system", "content": system_msgs[0]["text"]},
            {"role": "user", "content": user_msgs[0]["text"]},
        ]

        # --- Debug: dump full LLM input/output to file ---
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
        admissible_commands: list[str],
        last_action_error: str = "",
    ) -> tuple[str, dict]:
        """Get action and update UAM history.

        Returns:
            action: Validated action string (matched to admissible commands)
            agent_info: Dict with step_confidence, step_explanation, token usage
        """
        logger.info(f"Getting action for step {self.step_index}")

        system_msgs = self._build_system_messages()
        user_msgs = self._build_user_messages(
            goal, observation, admissible_commands, last_action_error
        )
        raw_text, step_usage = self._query_model(system_msgs, user_msgs)

        # Accumulate task-level token usage
        for k in self.task_agent_token_usage:
            self.task_agent_token_usage[k] += step_usage.get(k, 0)

        think = _parse_think(raw_text)
        action = _parse_action(raw_text)
        confidence = _parse_confidence(raw_text)
        explanation = _parse_explanation(raw_text)

        # Validate action against admissible commands
        validated_action = _validate_action(action, admissible_commands)
        if validated_action != action:
            logger.debug(
                f"  Action corrected: {action!r} -> {validated_action!r}"
            )

        logger.info(
            f"Step {self.step_index}: action={validated_action!r}, confidence={confidence:.4f}"
        )
        logger.debug(f"  think={think[:80]}...")
        logger.debug(f"  explanation={explanation[:80]}...")

        self.uam_history.append(
            {
                "step": self.step_index,
                "observation": observation,
                "think": think,
                "action": validated_action,
                "confidence": confidence,
                "explanation": explanation,
            }
        )
        self.step_index += 1

        return validated_action, {
            "step_confidence": confidence,
            "step_explanation": explanation,
            "step_agent_token_usage": step_usage,
            "task_agent_token_usage": dict(self.task_agent_token_usage),
        }


# ---------------------------------------------------------------------------
# TestRunner
# ---------------------------------------------------------------------------


class TestRunner:
    """Runs AUQ agent on ALfWorld benchmark tasks."""

    def __init__(
        self,
        model_name: str = "openrouter/openai/gpt-5.1",
        provider: str | None = None,
        include_obs_in_history: bool = True,
        results_dir: str = "./verification_results/alfworld/uam",
        n_tasks: int = 100,
        max_steps: int = 25,
        config_path: str = "benchmarks/alfworld/configs/base_config.yaml",
        task_types: list[int] | None = None,
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
        self.config_path = config_path
        self.task_types = task_types or list(TASK_TYPES.keys())
        self.shuffle_tasks = shuffle_tasks
        self.random_seed = random_seed

    def _init_alfworld_env(self) -> tuple[Any, Any]:
        """Initialize ALfWorld TextWorld environment.

        Returns:
            alfred_env: AlfredTWEnv instance (manages game files)
            env: Gym environment (for reset/step)
        """
        config = _load_alfworld_config(self.config_path)

        # Override config values for our use case
        config["env"]["task_types"] = self.task_types
        config["dagger"]["training"]["max_nb_steps_per_episode"] = self.max_steps

        # Use eval_out_of_distribution split
        alfred_env = get_environment("AlfredTWEnv")(
            config, train_eval="eval_out_of_distribution"
        )

        # Shuffle and limit game files before init_env
        if self.shuffle_tasks:
            random.shuffle(alfred_env.game_files)
            logger.info(
                f"Shuffled {len(alfred_env.game_files)} game files, "
                f"selecting up to {self.n_tasks}"
            )

        alfred_env.game_files = alfred_env.game_files[: self.n_tasks]
        alfred_env.num_games = len(alfred_env.game_files)

        env = alfred_env.init_env(batch_size=1)
        return alfred_env, env

    def run_agent(self) -> tuple[list[list[float]], list[int], dict[str, Any]]:
        """Run agent on ALfWorld tasks.

        Returns:
            all_step_confidences: Per-step confidence lists for each episode
            labels: Binary success labels (1=success, 0=failure)
            task_details: Per-task detail dict
        """
        logger.info(f"Running AUQ agent on ALfWorld: model={self.model_name}")

        alfred_env, env = self._init_alfworld_env()
        num_games = len(alfred_env.game_files)
        logger.info(f"Initialized env with {num_games} games")

        all_step_confidences: list[list[float]] = []
        labels: list[int] = []
        task_details: dict[str, Any] = {}

        # Load already-completed task results
        for task_file in sorted(self.tasks_dir.glob("task_*.json")):
            with open(task_file) as f:
                td = json.load(f)
            task_key = td["game_file"]
            task_details[task_key] = td
            all_step_confidences.append(td["step_confidences"])
            labels.append(td["success"])
            logger.info(f"Loaded existing result for {Path(task_key).name}")

        completed_games = set(task_details.keys())

        games_to_run = [
            gf for gf in alfred_env.game_files if gf not in completed_games
        ]
        logger.info(
            f"Games to run: {len(games_to_run)} "
            f"(skipping {len(completed_games)} already completed)"
        )

        for game_idx, game_file in enumerate(games_to_run):
            logger.info(f"--- Game {game_idx + 1}/{len(games_to_run)}: {Path(game_file).parent.name} ---")
            try:
                obs, infos = env.reset()
                observation = obs[0]
                game_file_actual = infos["extra.gamefile"][0]

                goal = _extract_goal(observation)
                task_type_name = _get_task_type_from_path(game_file_actual)
                task_type_id = _get_task_type_id(task_type_name) if task_type_name else None
                logger.info(f"  Goal: {goal[:100]}...")
                logger.info(f"  Task type: {task_type_name} (id={task_type_id})")

                agent = AUQAlfWorldAgent(
                    model_name=self.model_name, provider=self.provider,
                    include_obs_in_history=self.include_obs_in_history,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    debug_dir = self.results_dir / "debug_dumps"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    safe_name = re.sub(r"[^\w\-.]", "_", Path(game_file_actual).parent.name)
                    dump_path = debug_dir / f"{safe_name}.txt"
                    agent._debug_dump_path = str(dump_path)
                    agent._debug_goal_shown = goal
                    with open(dump_path, "w") as df:
                        df.write(f"Game file: {game_file_actual}\n")
                        df.write(f"Goal: {goal}\n")
                        df.write(f"Initial observation: {observation}\n")

                step_confidences: list[float] = []
                won = False

                for step in range(self.max_steps):
                    admissible_commands = infos["admissible_commands"][0]
                    action, info = agent.get_action(
                        goal=goal,
                        observation=observation,
                        admissible_commands=admissible_commands,
                        last_action_error="",
                    )
                    step_confidences.append(info["step_confidence"])

                    obs, scores, dones, infos = env.step([action])
                    observation = obs[0]
                    done = dones[0]
                    won = infos["won"][0]

                    logger.debug(
                        f"  step={step}, action={action!r}, "
                        f"done={done}, won={won}"
                    )

                    if done:
                        break

                success = 1 if won else 0
                all_step_confidences.append(step_confidences)
                labels.append(success)

                task_key = game_file_actual
                task_details[task_key] = {
                    "game_file": game_file_actual,
                    "goal": goal,
                    "task_type": task_type_name,
                    "task_type_id": task_type_id,
                    "step_confidences": step_confidences,
                    "success": success,
                    "task_agent_token_usage": info.get("task_agent_token_usage", {}),
                    "n_steps": len(step_confidences),
                }

                # Save per-task result for resumability
                task_file_name = re.sub(r"[^\w\-.]", "_", Path(game_file_actual).parent.name)
                task_file = self.tasks_dir / f"task_{task_file_name}.json"
                with open(task_file, "w") as f:
                    json.dump(task_details[task_key], f, indent=2)
                logger.debug(f"  Saved result to {task_file}")

                logger.info(
                    f"  Result: steps={len(step_confidences)}, "
                    f"success={success}, "
                    f"tokens={info.get('task_agent_token_usage', {}).get('total_tokens', '?')}"
                )

            except Exception as e:
                logger.error(f"Error on game {game_file}: {e}", exc_info=True)

        env.close()
        logger.info(f"Completed {len(labels)} episodes")
        return all_step_confidences, labels, task_details

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
        """Compute AUROC, ECE, and Brier Score for all four aggregation methods.

        Paper Section 3.4:
            Φ_last    (End-State Belief):    C(τ) = ĉ_T
            Φ_avg     (Overall Quality):     C(τ) = (1/T) Σ ĉ_t
            Φ_min     (Process Reliability): C(τ) = min ĉ_t
            Φ_product (Chain Probability):   C(τ) = Π ĉ_t

        Returns dict keyed by method name, each with auroc/ece/brier keys.
        """
        fallback = {"auroc": 0.5, "ece": 1.0, "brier": 1.0}
        if len(set(labels)) < 2:
            logger.warning("Only one class present, returning fallback metrics")
            return {m: fallback.copy() for m in ("last", "avg", "min", "product")}

        results: dict[str, dict[str, float]] = {}
        for method in ("last", "avg", "min", "product"):
            traj_confs = [
                TestRunner._aggregate(confs, method) for confs in all_step_confidences
            ]
            try:
                auroc = roc_auc_score(labels, traj_confs)
                ece = TestRunner._ece(traj_confs, labels)
                brier = brier_score_loss(labels, traj_confs)
                results[method] = {"auroc": auroc, "ece": ece, "brier": brier}
                logger.info(
                    f"  Φ_{method}: AUROC={auroc:.4f}, ECE={ece:.4f}, Brier={brier:.4f}"
                )
            except Exception as e:
                logger.error(f"Error computing metrics for Φ_{method}: {e}")
                results[method] = fallback.copy()

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="AUQ verification on ALfWorld")
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
        "--n_tasks", type=int, default=100, help="Number of ALfWorld tasks to run"
    )
    parser.add_argument(
        "--max_steps", type=int, default=25, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--config_path",
        default="benchmarks/alfworld/configs/base_config.yaml",
        help="Path to ALfWorld config YAML",
    )
    parser.add_argument(
        "--task_types",
        type=int,
        nargs="+",
        default=None,
        choices=[1, 2, 3, 4, 5, 6],
        help="Task type IDs to include (1-6, default: all)",
    )
    parser.add_argument(
        "--results_dir",
        default="./verification_results/alfworld/uam",
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
        help="Randomly shuffle tasks before selection",
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
    logger.info("AUQ Verification on ALfWorld")
    logger.info(
        f"Model: {args.model}, Tasks: {args.n_tasks}, "
        f"obs_in_history: {args.include_obs_in_history}"
    )
    task_type_desc = (
        ", ".join(f"{t}={TASK_TYPE_LABELS.get(t, '?')}" for t in (args.task_types or list(TASK_TYPES.keys())))
    )
    logger.info(f"Task types: {task_type_desc}")
    logger.info("=" * 60)

    runner = TestRunner(
        model_name=args.model, provider=args.provider,
        include_obs_in_history=args.include_obs_in_history,
        results_dir=args.results_dir,
        n_tasks=args.n_tasks,
        max_steps=args.max_steps,
        config_path=args.config_path,
        task_types=args.task_types,
        shuffle_tasks=args.shuffle_tasks,
        random_seed=args.random_seed,
    )

    all_step_confidences, labels, task_details = runner.run_agent()

    if not labels:
        logger.error("No confidence values extracted, cannot compute metrics")
        return

    metrics = runner.compute_metrics(all_step_confidences, labels)
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

    # Per-task-type success breakdown
    type_results: dict[str, list[int]] = {}
    for detail in task_details.values():
        tt = detail.get("task_type", "unknown")
        type_results.setdefault(tt, []).append(detail["success"])
    if len(type_results) > 1:
        logger.info("  Per-task-type success rates:")
        for tt, successes in sorted(type_results.items()):
            tt_id = _get_task_type_id(tt) if tt else None
            label = TASK_TYPE_LABELS.get(tt_id, tt) if tt_id else tt
            logger.info(
                f"    {label}: {np.mean(successes):.2%} ({sum(successes)}/{len(successes)})"
            )
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
        "config_path": args.config_path,
        "task_types": args.task_types or list(TASK_TYPES.keys()),
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
