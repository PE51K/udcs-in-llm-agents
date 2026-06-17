#!/usr/bin/env python3
"""Proposed v4 Clarification Benchmark on ALFWorld
==================================================

Extends AUQ (Huang et al.) with a separate request uncertainty (u_request) source.
50% of tasks are underspecified (object or receptacle dropped from the goal using
traj_data.json). The agent has a request_clarification action to request a more
specified goal. If the agent requests clarification on an underspecified task,
the full goal is revealed and the episode continues. Evaluates clarification
precision/recall/F1/accuracy alongside standard fault-detection metrics
(confidence and u_request).

Run from project root:
    export OPENAI_API_KEY=$OPENAI_API_KEY
    export ALFWORLD_DATA=<path>
    uv run --group scripts python -m \
        scripts.verification.alfworld_clarification.proposed_v4.proposed_v4 \
        --n_tasks 100 --log_level INFO --random_seed 42

Arguments:
  --model               LLM model name (default: openrouter/openai/gpt-5.1)
  --primary_aggregation Method highlighted in logs: last|avg|min|product (default: avg)
  --n_tasks             Number of ALfWorld tasks to run (default: 100)
  --max_steps           Maximum steps per episode (default: 25)
  --config_path         Path to ALfWorld config YAML (default: benchmarks/alfworld/configs/base_config.yaml)
  --task_types          Task type IDs to include: 1-6 (default: all)
  --results_dir         Output directory (default: ./verification_results/alfworld_clarification/proposed_v4)
  --log_level           Logging level: DEBUG|INFO|WARNING|ERROR|CRITICAL (default: INFO)
  --random_seed         Random seed for reproducibility (default: None)
  --shuffle_tasks       Randomly shuffle tasks before selection (default: True)
  --include_obs_in_history  Include observation summaries in UAM history (default: True)
  --no-include_obs_in_history  Disable observations in UAM history

Output:
  test_results.json with per-method metrics (auroc/ece/brier), clarification
  metrics (precision/recall/f1/accuracy), and per-task details
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
# Clarification helpers
# ---------------------------------------------------------------------------


def _read_traj_data(game_file_path: str) -> dict:
    game_dir = Path(game_file_path).parent
    traj_file = game_dir / 'traj_data.json'
    with open(traj_file) as f:
        return json.load(f)


def _make_underspecified_alfworld(traj_data: dict, drop_object: bool) -> str:
    task_type = traj_data['task_type']
    params = traj_data['pddl_params']
    obj = params.get('object_target', 'object').lower()
    recep = params.get('parent_target', 'container').lower()
    toggle = params.get('toggle_target', 'light').lower()
    if task_type == 'look_at_obj_in_light':
        return f"examine something with the {toggle}" if drop_object else f"examine the {obj} somehow"
    if task_type == 'pick_two_obj_and_place':
        return f"put two things in {recep}" if drop_object else f"put two {obj} somewhere"
    return f"put something in {recep}" if drop_object else f"put a {obj} somewhere"


def _build_clarification_task_set(n_tasks: int, random_seed: int | None = None) -> list[tuple[int, bool]]:
    rng = random.Random(random_seed)
    labels = [0] * (n_tasks // 2) + [1] * (n_tasks - n_tasks // 2)
    drop_choices = [rng.choice([True, False]) for _ in range(n_tasks)]
    combined = list(zip(labels, drop_choices))
    rng.shuffle(combined)
    return combined


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ProposedV4AlfWorldAgent:
    """Proposed v4 agent for ALfWorld: UAM confidence + u_request decomposition.

    Extends AUQ (Huang et al.) with a separate request uncertainty (u_request) source.
    The only difference from AUQ is:
    - Elicits u_request (0.0-1.0) and u_request_explanation per step
    - If u_request >= clarification_threshold, the agent must choose request_clarification
    - u_request and u_request_explanation are propagated in UAM history
    """

    def __init__(
        self,
        model_name: str = "openrouter/openai/gpt-5.1",
        temperature: float = 0.0,
        provider: str | None = None,
        include_obs_in_history: bool = True,
        clarification_threshold: float = 0.5,
    ) -> None:
        self.include_obs_in_history = include_obs_in_history
        self.temperature = temperature
        self.clarification_threshold = clarification_threshold

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
        """Build system message following the paper's baseline prompt structure."""
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
        """Build user messages following the paper's prompt template.

        Structure:
        - Task description (goal)
        - Step count + UAM history (Variant B: obs, think, u_request, action, confidence, explanation)
        - Current step + observation + admissible actions
        - ReAct instruction with u_request assessment
        - Confidence Elicitation Suffix (identical to AUQ)
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
                lines.append(f"<think>{entry['think']}</think>")
                lines.append(f"<u_request>{entry['u_request']}</u_request>")
                lines.append(f"<u_request_explanation>{entry['u_request_explanation']}</u_request_explanation>")
                lines.append(f"<action>{entry['action']}</action>")
                lines.append(f"<confidence>{entry['confidence']}</confidence>")
                lines.append(f"<explanation>{entry['explanation']}</explanation>")
                parts.append("\n".join(lines))

        # --- Last Action Error ---
        if last_action_error:
            parts.append(f"Error from last action: {last_action_error}")

        # --- Current step + observation + admissible actions ---
        admissible_str = ", ".join(admissible_commands) if admissible_commands else "(none)"
        admissible_str += ", request_clarification (request a more specified goal if the task is missing key details or has multiple valid solutions)"
        parts.append(
            f"You are now at step {self.step_index} and your current observation is: "
            f"{observation}\n"
            f"Your admissible actions of the current situation are: [{admissible_str}]."
        )

        # --- ReAct instruction with u_request assessment ---
        parts.append(
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
            f"If u_request >= {self.clarification_threshold}, your action MUST be 'request_clarification'.\n"
            "Once you've finished your reasoning, you should choose an admissible "
            "action for the current step and present it within <action> </action> tags."
        )

        # --- Confidence Elicitation Suffix (identical to AUQ) ---
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

        return [{"type": "text", "text": "\n".join(parts)}]

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
        admissible_commands: list[str],
        last_action_error: str = "",
    ) -> tuple[str, dict]:
        """Get action and update UAM history.

        Returns:
            action: Validated action string (matched to admissible commands)
            agent_info: Dict with step_confidence, step_u_request, step_explanation, token usage
        """
        logger.info(f"Getting action for step {self.step_index}")

        system_msgs = self._build_system_messages()
        user_msgs = self._build_user_messages(
            goal, observation, admissible_commands, last_action_error
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

        # Validate action against admissible commands
        validated_action = _validate_action(action, admissible_commands)
        if validated_action != action:
            logger.debug(
                f"  Action corrected: {action!r} -> {validated_action!r}"
            )

        logger.info(
            f"Step {self.step_index}: action={validated_action!r}, confidence={confidence:.4f}, u_request={u_request:.4f}"
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
                "action": validated_action,
                "confidence": confidence,
                "explanation": explanation,
            }
        )
        self.step_index += 1

        return validated_action, {
            "step_confidence": confidence,
            "step_u_request": u_request,
            "step_explanation": explanation,
            "step_think": think,
            "step_action": validated_action,
            "step_u_request_explanation": u_request_explanation,
            "step_agent_token_usage": step_usage,
            "task_agent_token_usage": dict(self.task_agent_token_usage),
        }


# ---------------------------------------------------------------------------
# TestRunner
# ---------------------------------------------------------------------------


class TestRunner:
    """Runs Proposed v4 agent on ALfWorld clarification benchmark."""

    def __init__(self, model_name="openrouter/openai/gpt-5.1", provider=None, include_obs_in_history=True,
                 results_dir="./verification_results/alfworld_clarification/proposed_v4", n_tasks=100,
                 max_steps=25, config_path="benchmarks/alfworld/configs/base_config.yaml",
                 task_types=None, shuffle_tasks=True, random_seed=None,
                 clarification_threshold=0.5):
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
        self.clarification_threshold = clarification_threshold

    def _init_alfworld_env(self) -> tuple[Any, Any]:
        config = _load_alfworld_config(self.config_path)
        config["env"]["task_types"] = self.task_types
        config["dagger"]["training"]["max_nb_steps_per_episode"] = self.max_steps
        alfred_env = get_environment("AlfredTWEnv")(config, train_eval="eval_out_of_distribution")
        if self.shuffle_tasks:
            random.shuffle(alfred_env.game_files)
        alfred_env.game_files = alfred_env.game_files[: self.n_tasks]
        alfred_env.num_games = len(alfred_env.game_files)
        env = alfred_env.init_env(batch_size=1)
        return alfred_env, env

    def run_agent(self) -> tuple[list[list[float]], list[list[float]], list[int], list[int], list[int], dict[str, Any]]:
        """Run agent on ALfWorld clarification benchmark."""
        logger.info(f"Running proposed v4 clarification agent on ALfWorld: model={self.model_name}")
        alfred_env, env = self._init_alfworld_env()
        num_games = len(alfred_env.game_files)
        clarification_set = _build_clarification_task_set(num_games, self.random_seed)

        all_step_confidences: list[list[float]] = []
        all_step_u_requests: list[list[float]] = []
        labels: list[int] = []
        task_labels: list[int] = []
        all_agent_asked: list[int] = []
        task_details: dict[str, Any] = {}

        for task_file in sorted(self.tasks_dir.glob("task_*.json")):
            with open(task_file) as f:
                td = json.load(f)
            task_details[td["game_file"]] = td
            all_step_confidences.append(td["step_confidences"])
            all_step_u_requests.append(td.get("step_u_requests", [0.0] * len(td["step_confidences"])))
            labels.append(td["success"])
            task_labels.append(td["task_label"])
            all_agent_asked.append(td["agent_asked"])

        completed_games = set(task_details.keys())

        for game_idx, game_file in enumerate(alfred_env.game_files):
            if game_file in completed_games:
                continue
            task_label, drop_object = clarification_set[game_idx]
            logger.info(f"--- Game {game_idx+1}/{num_games}: {Path(game_file).parent.name} "
                        f"({'underspecified' if task_label else 'specified'}) ---")
            try:
                obs, infos = env.reset()
                observation = obs[0]
                game_file_actual = infos["extra.gamefile"][0]
                original_goal = _extract_goal(observation)
                task_type_name = _get_task_type_from_path(game_file_actual)
                task_type_id = _get_task_type_id(task_type_name) if task_type_name else None

                if task_label == 1:
                    try:
                        traj_data = _read_traj_data(game_file_actual)
                        goal = _make_underspecified_alfworld(traj_data, drop_object)
                    except Exception as e:
                        logger.warning(f"  Could not underspecify: {e}")
                        goal = original_goal
                        task_label = 0
                    goal_shown = goal  # save before potential reveal
                    # Strip original goal from observation so it doesn't leak
                    observation = re.sub(
                        r"Your task is to:\s*.+",
                        f"Your task is to: {goal}",
                        observation,
                    )
                    if original_goal in observation:
                        raise RuntimeError(f"LEAK: original goal still in initial obs after sanitization ({Path(game_file_actual).parent.name})")
                    else:
                        logger.debug(f"  Sanitized initial obs: replaced original goal")
                    logger.info(f"  Original: {original_goal[:100]}...")
                    logger.info(f"  Underspecified: {goal[:100]}...")
                else:
                    goal = original_goal
                    goal_shown = goal
                    logger.info(f"  Goal: {goal[:100]}...")

                agent = ProposedV4AlfWorldAgent(model_name=self.model_name, provider=self.provider,
                                                 include_obs_in_history=self.include_obs_in_history,
                                                 clarification_threshold=self.clarification_threshold)
                if logger.isEnabledFor(logging.DEBUG):
                    debug_dir = self.results_dir / "debug_dumps"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    safe_name = re.sub(r"[^\w\-.]", "_", Path(game_file_actual).parent.name)
                    dump_path = debug_dir / f"{safe_name}_label{task_label}.txt"
                    agent._debug_dump_path = str(dump_path)
                    agent._debug_goal_shown = goal_shown if task_label == 1 else goal
                    with open(dump_path, "w") as df:
                        df.write(f"Game file: {game_file_actual}\n")
                        df.write(f"Task label: {task_label} ({'underspecified' if task_label else 'specified'})\n")
                        df.write(f"Original goal: {original_goal}\n")
                        df.write(f"Goal shown to agent: {goal_shown if task_label == 1 else goal}\n")
                        df.write(f"Initial observation: {observation}\n")

                step_confidences: list[float] = []
                step_u_requests: list[float] = []
                step_traces: list[dict] = []
                won = False
                agent_asked = 0

                for step in range(self.max_steps):
                    admissible_commands = infos["admissible_commands"][0]
                    action, info = agent.get_action(goal=goal, observation=observation,
                                                     admissible_commands=admissible_commands)
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

                    if action.lower().startswith("request_clarification"):
                        agent_asked = 1
                        logger.info(f"  Step {step}: agent requested clarification")
                        if task_label == 1:
                            goal = original_goal
                            logger.info(f"  Revealed full goal: {original_goal[:100]}...")
                        continue

                    obs, scores, dones, infos = env.step([action])
                    observation = obs[0]
                    done = dones[0]
                    won = infos["won"][0]
                    if done:
                        break

                success = 1 if won else 0
                all_step_confidences.append(step_confidences)
                all_step_u_requests.append(step_u_requests)
                labels.append(success)
                task_labels.append(task_label)
                all_agent_asked.append(agent_asked)

                task_details[game_file_actual] = {
                    "game_file": game_file_actual, "goal": original_goal, "goal_shown": goal_shown,
                    "task_label": task_label, "agent_asked": agent_asked,
                    "drop_object": drop_object if task_label == 1 else None,
                    "task_type": task_type_name, "task_type_id": task_type_id,
                    "step_confidences": step_confidences, "step_u_requests": step_u_requests,
                    "step_traces": step_traces, "success": success,
                    "task_agent_token_usage": info.get("task_agent_token_usage", {}),
                    "n_steps": len(step_confidences),
                }
                task_file_name = re.sub(r"[^\w\-.]", "_", Path(game_file_actual).parent.name)
                with open(self.tasks_dir / f"task_{task_file_name}.json", "w") as f:
                    json.dump(task_details[game_file_actual], f, indent=2)
                logger.info(f"  Result: steps={len(step_confidences)}, success={success}, asked={agent_asked}")
            except Exception as e:
                logger.error(f"Error on game {game_file}: {e}", exc_info=True)

        env.close()
        return all_step_confidences, all_step_u_requests, labels, task_labels, all_agent_asked, task_details

    @staticmethod
    def _aggregate(confs, method):
        if method == "last": return confs[-1]
        if method == "avg": return float(np.mean(confs))
        if method == "min": return float(np.min(confs))
        if method == "product": return float(np.exp(np.sum(np.log(np.clip(confs, 1e-9, 1.0)))))
        raise ValueError(f"Unknown aggregation: {method!r}")

    @staticmethod
    def _aggregate_u_request(scores, method):
        if method == "first": return scores[0]
        if method == "max": return float(np.max(scores))
        if method == "avg": return float(np.mean(scores))
        if method == "product": return float(1.0 - np.exp(np.sum(np.log(np.clip([1.0 - s for s in scores], 1e-9, 1.0)))))
        raise ValueError(f"Unknown u_request aggregation: {method!r}")

    @staticmethod
    def _ece(confidences, labels, n_bins=10):
        confs = np.array(confidences)
        labs = np.array(labels, dtype=float)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        n = len(confs)
        ece = 0.0
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            mask = (confs >= lo) & (confs <= hi) if i == n_bins - 1 else (confs >= lo) & (confs < hi)
            if mask.sum() == 0: continue
            ece += mask.sum() * abs(confs[mask].mean() - labs[mask].mean())
        return float(ece / n)

    @staticmethod
    def compute_metrics(all_step_confidences, all_step_u_requests, labels):
        fallback = {"auroc": 0.5, "ece": 1.0, "brier": 1.0}
        if len(set(labels)) < 2:
            keys = [f"confidence/{m}" for m in ("last", "avg", "min", "product")]
            keys += [f"u_request/{m}" for m in ("first", "max", "avg", "product")]
            return {k: fallback.copy() for k in keys}
        results = {}
        for method in ("last", "avg", "min", "product"):
            traj_confs = [TestRunner._aggregate(c, method) for c in all_step_confidences]
            try:
                results[f"confidence/{method}"] = {
                    "auroc": roc_auc_score(labels, traj_confs),
                    "ece": TestRunner._ece(traj_confs, labels),
                    "brier": brier_score_loss(labels, traj_confs),
                }
            except Exception:
                results[f"confidence/{method}"] = fallback.copy()
        # u_request aggregations (inverted: high u_request = low confidence in success)
        for method in ("first", "max", "avg", "product"):
            traj_u_reqs = [TestRunner._aggregate_u_request(r, method) for r in all_step_u_requests]
            traj_confs_from_u = [1.0 - u for u in traj_u_reqs]
            try:
                results[f"u_request/{method}"] = {
                    "auroc": roc_auc_score(labels, traj_confs_from_u),
                    "ece": TestRunner._ece(traj_confs_from_u, labels),
                    "brier": brier_score_loss(labels, traj_confs_from_u),
                }
            except Exception:
                results[f"u_request/{method}"] = fallback.copy()
        return results

    @staticmethod
    def compute_clarification_metrics(all_agent_asked, task_labels):
        tp = sum(1 for a, l in zip(all_agent_asked, task_labels) if a == 1 and l == 1)
        fp = sum(1 for a, l in zip(all_agent_asked, task_labels) if a == 1 and l == 0)
        fn = sum(1 for a, l in zip(all_agent_asked, task_labels) if a == 0 and l == 1)
        tn = sum(1 for a, l in zip(all_agent_asked, task_labels) if a == 0 and l == 0)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        acc = (tp + tn) / len(task_labels) if task_labels else 0.0
        return {"precision": p, "recall": r, "f1": f1, "accuracy": acc}


def main():
    parser = argparse.ArgumentParser(description="Proposed v4 clarification benchmark on ALfWorld")
    parser.add_argument("--model", default=os.getenv("MODEL", "openrouter/openai/gpt-5.1"))
    parser.add_argument("--primary_aggregation", default="avg", choices=["last", "avg", "min", "product"])
    parser.add_argument("--n_tasks", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=25)
    parser.add_argument("--config_path", default="benchmarks/alfworld/configs/base_config.yaml")
    parser.add_argument("--task_types", type=int, nargs="+", default=None, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--results_dir", default="./verification_results/alfworld_clarification/proposed_v4")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--random_seed", type=int, default=None)
    parser.add_argument("--shuffle_tasks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_obs_in_history", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clarification_threshold", type=float, default=0.5,
                        help="u_request threshold for requesting clarification")
    parser.add_argument(
        "--provider",
        default=os.getenv("PROVIDER"),
        help="OpenRouter provider slug(s) to pin, comma-separated for an ordered allowlist (e.g. DeepInfra,Parasail); disables fallback. Default none (auto-route).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    for lg in ("openai", "httpx", "urllib3"):
        logging.getLogger(lg).setLevel(logging.WARNING)
    log_level = getattr(logging, args.log_level)
    logging.getLogger("__main__").setLevel(log_level)
    logging.getLogger("scripts").setLevel(log_level)

    if args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)

    logger.info("=" * 60)
    logger.info("Proposed v4 Clarification Benchmark on ALfWorld")
    logger.info(f"Model: {args.model}, Tasks: {args.n_tasks}")
    logger.info("=" * 60)

    runner = TestRunner(model_name=args.model, provider=args.provider, include_obs_in_history=args.include_obs_in_history,
                        results_dir=args.results_dir, n_tasks=args.n_tasks, max_steps=args.max_steps,
                        config_path=args.config_path, task_types=args.task_types,
                        shuffle_tasks=args.shuffle_tasks, random_seed=args.random_seed,
                        clarification_threshold=args.clarification_threshold)

    all_step_confidences, all_step_u_requests, labels, task_labels, all_agent_asked, task_details = runner.run_agent()
    if not labels:
        logger.error("No results"); return

    metrics = runner.compute_metrics(all_step_confidences, all_step_u_requests, labels)
    clarification_metrics = runner.compute_clarification_metrics(all_agent_asked, task_labels)
    success_rate = float(np.mean(labels))
    n_spec = sum(1 for l in task_labels if l == 0)
    n_under = sum(1 for l in task_labels if l == 1)

    logger.info("=" * 60)
    for method, m in metrics.items():
        logger.info(f"  {method}: AUROC={m['auroc']:.4f}  ECE={m['ece']:.4f}  Brier={m['brier']:.4f}")
    logger.info(f"  Success Rate: {success_rate:.2%}")
    logger.info(f"  Tasks: {len(labels)} ({n_spec} specified, {n_under} underspecified)")
    logger.info(f"  Clarification: P={clarification_metrics['precision']:.4f} "
                f"R={clarification_metrics['recall']:.4f} F1={clarification_metrics['f1']:.4f} "
                f"Acc={clarification_metrics['accuracy']:.4f}")
    logger.info("=" * 60)

    for detail in task_details.values():
        confs = detail["step_confidences"]
        for method in ("last", "avg", "min", "product"):
            detail[f"c_{method}"] = TestRunner._aggregate(confs, method)

    test_results = {
        "settings": vars(args),
        "model": args.model, "primary_aggregation": args.primary_aggregation,
        "n_tasks": args.n_tasks, "max_steps": args.max_steps,
        "config_path": args.config_path, "task_types": args.task_types or list(TASK_TYPES.keys()),
        "clarification_threshold": args.clarification_threshold,
        "random_seed": args.random_seed, "success_rate": success_rate,
        "tasks_evaluated": len(labels), "n_specified": n_spec, "n_underspecified": n_under,
        "metrics": metrics, "clarification_metrics": clarification_metrics,
        "task_details": task_details,
    }
    output_path = Path(args.results_dir) / "test_results.json"
    with open(output_path, "w") as f:
        json.dump(test_results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
