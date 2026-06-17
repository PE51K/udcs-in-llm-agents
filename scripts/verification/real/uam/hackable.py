#!/usr/bin/env python3

"""AUQ (Agentic Uncertainty Quantification) hackable agent.

Standalone implementation of System 1: Forward UQ via Uncertainty-Aware Memory (UAM)
from "Agentic Uncertainty Quantification" by Huang et al.

No guawa dependency. Parses verbalized confidence directly from structured
<think>/<action>/<confidence>/<explanation> output.

Memory structure (paper Eq. 3):
    M_t = {(o_i, a_i, ĉ_i, ê_i)}_{i=0}^{t-1}

Observations in history are optional (include_obs_in_history flag).
"""

import base64
import io
import logging
import os
import re
from typing import Literal

import numpy as np
from agisdk.REAL.browsergym.core.action.highlevel import HighLevelActionSet
from agisdk.REAL.browsergym.experiments import Agent
from agisdk.REAL.browsergym.utils.obs import (
    flatten_axtree_to_str,
    flatten_dom_to_str,
    prune_html,
)
from openai import OpenAI
from PIL import Image

logger = logging.getLogger(__name__)


def image_to_jpg_base64_url(image: np.ndarray | Image.Image) -> str:
    """Convert image to base64 encoded JPEG URL."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{image_base64}"


def _parse_think(text: str) -> str:
    """Extract content from <think> tags. Returns empty string if absent."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_action(text: str) -> str:
    """Extract action from <action> tags. Falls back to backtick code block, then full text."""
    match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(?:\w+)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_confidence(text: str) -> float:
    """Parse confidence from <confidence> tags, clamped to [0, 1].

    Raises:
        ValueError: If <confidence> tag is missing or unparseable.
    """
    match = re.search(r"<confidence>\s*([\d.]+)\s*</confidence>", text)
    if not match:
        raise ValueError(
            f"AUQ: missing <confidence> tag in agent output. Output (first 200 chars): {text[:200]}"
        )
    return max(0.0, min(1.0, float(match.group(1))))


def _parse_explanation(text: str) -> str:
    """Extract content from <explanation> tags. Returns empty string if absent."""
    match = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


class AUQHackableAgent(Agent):
    """Hackable agent implementing Forward UQ via Uncertainty-Aware Memory (UAM).

    Implements System 1 from the AUQ paper (Huang et al.):
    - Confidence Elicitation Protocol: structured <think>/<action>/<confidence>/<explanation>
    - Uncertainty-Aware Memory (UAM): past actions, confidence scores, and explanations
      are propagated into the prompt (Variant B: Semantic Propagation)
    - Observations in history are optional via include_obs_in_history flag
    """

    def obs_preprocessor(self, obs: dict) -> dict:
        return {
            "chat_messages": obs["chat_messages"],
            "screenshot": obs["screenshot"],
            "goal_object": obs["goal_object"],
            "last_action": obs["last_action"],
            "last_action_error": obs["last_action_error"],
            "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
            "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
        }

    def __init__(
        self,
        model_name: str,
        chat_mode: bool = False,
        demo_mode: str = "off",
        use_html: bool = False,
        use_axtree: bool = True,
        use_screenshot: bool = False,
        temperature: float | None = None,
        provider: str | None = None,
        include_obs_in_history: bool = True,
        system_message_handling: Literal["separate", "combined"] = "separate",
    ) -> None:
        super().__init__()
        self.chat_mode = chat_mode
        self.use_html = use_html
        self.use_axtree = use_axtree
        self.use_screenshot = use_screenshot
        self.temperature = temperature
        self.provider = provider
        self.include_obs_in_history = include_obs_in_history
        self.system_message_handling = system_message_handling

        if not (use_html or use_axtree):
            raise ValueError("Either use_html or use_axtree must be set to True.")

        actual_model_name = model_name.replace("openrouter/", "", 1)
        self.client = OpenAI(
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.model_name = actual_model_name
        self.is_openrouter = True

        self.action_set = HighLevelActionSet(
            subsets=["chat", "bid", "infeas"],
            strict=False,
            multiaction=False,
            demo_mode=demo_mode,
        )
        self.step_index = 0
        self.uam_history: list[dict] = []
        # Cumulative token usage for this task
        self.task_agent_token_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def _api_params(self) -> dict:
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
        return params

    def _build_system_messages(self) -> list[dict]:
        """System message with <think>/<action> format instruction (paper Appendix)."""
        if self.chat_mode:
            base = (
                "# Instructions\n\n"
                "You are a UI Assistant, your goal is to help the user perform tasks "
                "using a web browser. You can communicate with the user via a chat, "
                "to which the user gives you instructions and to which you can send "
                "back messages. You have access to a web browser that both you and "
                "the user can see, and with which only you can interact via specific "
                "commands.\n\n"
                "Review the instructions from the user, the current state of the page "
                "and all other information to find the best possible next action to "
                "accomplish your goal. Your answer will be interpreted and executed by "
                "a program, make sure to follow the formatting instructions.\n\n"
            )
        else:
            base = (
                "# Instructions\n\n"
                "Review the current state of the page and all other information to find "
                "the best possible next action to accomplish your goal. Your answer will "
                "be interpreted and executed by a program, make sure to follow the "
                "formatting instructions.\n\n"
            )
        base += (
            "You should first reason step-by-step about the current situation. "
            "This reasoning process MUST be enclosed within <think> </think> tags.\n"
            "Once you've finished your reasoning, you should choose an admissible "
            "action for the current step and present it within <action> </action> tags."
        )
        return [{"type": "text", "text": base}]

    def _build_user_messages(self, obs: dict) -> list[dict]:
        """Build user messages with paper-exact UAM format and elicitation suffix.

        Structure:
        1. Goal / Chat messages
        2. Current observation (AXTree, DOM, Screenshot)
        3. Action Space
        4. UAM History (paper Variant B: each step has obs?, think, action, confidence, explanation)
        5. Error from last action (if any)
        6. Next action instruction (with step count)
        7. Confidence Elicitation Suffix
        """
        msgs = []

        # --- Goal / Chat ---
        if self.chat_mode:
            msgs.append({"type": "text", "text": "# Chat Messages"})
            for msg in obs["chat_messages"]:
                if msg["role"] in ("user", "assistant", "infeasible"):
                    msgs.append(
                        {"type": "text", "text": f"- [{msg['role']}] {msg['message']}"}
                    )
                elif msg["role"] == "user_image":
                    msgs.append({"type": "image_url", "image_url": msg["message"]})
                else:
                    raise ValueError(f"Unexpected chat message role {msg['role']!r}")
        else:
            assert obs["goal_object"], "The goal is missing."
            msgs.append({"type": "text", "text": "# Goal"})
            msgs.extend(obs["goal_object"])

        # --- Current Observation ---
        if self.use_axtree:
            msgs.append(
                {
                    "type": "text",
                    "text": f"# Current page Accessibility Tree\n\n{obs['axtree_txt']}",
                }
            )
        if self.use_html:
            msgs.append(
                {"type": "text", "text": f"# Current page DOM\n\n{obs['pruned_html']}"}
            )
        if self.use_screenshot:
            msgs.extend(
                [
                    {"type": "text", "text": "# Current page Screenshot"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_jpg_base64_url(obs["screenshot"]),
                            "detail": "auto",
                        },
                    },
                ]
            )

        # --- Action Space ---
        msgs.append(
            {
                "type": "text",
                "text": (
                    f"# Action Space\n\n"
                    f"{self.action_set.describe(with_long_description=False, with_examples=True)}"
                ),
            }
        )

        # --- UAM History (paper Variant B: Semantic Propagation) ---
        # M_t = {(o_i, a_i, ĉ_i, ê_i)}_{i=0}^{t-1}
        if self.uam_history:
            msgs.append({"type": "text", "text": "# History of past actions"})
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
                msgs.append({"type": "text", "text": "\n".join(lines)})

            if obs["last_action_error"]:
                msgs.append(
                    {
                        "type": "text",
                        "text": f"# Error message from last action\n\n{obs['last_action_error']}",
                    }
                )

        # --- Next Action (with step count, matching paper) ---
        msgs.append(
            {
                "type": "text",
                "text": (
                    "# Next action\n\n"
                    f"You are now at step {self.step_index}. "
                    f"Prior to this step, you have already taken {self.step_index} step(s).\n\n"
                    "Now it's your turn to take an action."
                ),
            }
        )

        # --- Confidence Elicitation Suffix (paper Appendix) ---
        msgs.append(
            {
                "type": "text",
                "text": (
                    "After your action, you MUST provide:\n\n"
                    "1. Your confidence level (0.0-1.0) in <confidence>...</confidence> tags\n\n"
                    "2. An explanation of your confidence in <explanation>...</explanation> tags\n"
                    "   - Explain what makes you confident\n"
                    "   - Explain what concerns or uncertainties you have\n"
                    "   - What information might be missing or unclear\n"
                    "   - What alternative actions you considered\n"
                    "   - DO NOT output empty <explanation></explanation> tags - "
                    "you MUST provide actual text inside"
                ),
            }
        )

        return msgs

    def _query_model(self, system_msgs: list[dict], user_msgs: list[dict]) -> tuple[str, dict]:
        """Single API call, returns (raw_text, usage_dict)."""
        params = self._api_params()
        if self.system_message_handling == "combined":
            combined = system_msgs[0]["text"] + "\n\n"
            for msg in user_msgs:
                if msg["type"] == "text":
                    combined += msg["text"] + "\n"
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": combined}],
                n=1,
                **params,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_msgs[0]["text"]},
                    {"role": "user", "content": user_msgs},
                ],
                n=1,
                **params,
            )
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
        # OpenRouter routes some providers' tagged output into `reasoning`;
        # reconstruct the full text before parsing (content may be None/empty).
        _msg = response.choices[0].message
        _reasoning = getattr(_msg, "reasoning", None) or getattr(_msg, "reasoning_content", None) or ""
        result_text = (_reasoning + "\n" + (_msg.content or "")).strip() if _reasoning else (_msg.content or "")
        return result_text, usage

    def get_action(self, obs: dict) -> tuple[str, dict]:
        """Get action and update UAM history.

        Paper Algorithm 1, Phase 3 (Memory Update):
            M_{t+1} ← M_t ∪ {(o_t, a_t, ĉ_t, ê_t)}

        Returns:
            action: Parsed action string
            agent_info: Dict with step_confidence and step_explanation
        """
        logger.info(f"Getting action for step {self.step_index}")

        obs_summary = obs.get("axtree_txt", "")
        system_msgs = self._build_system_messages()
        user_msgs = self._build_user_messages(obs)
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
        logger.debug(f"  explanation={explanation[:80]}...")

        self.uam_history.append(
            {
                "step": self.step_index,
                "observation": obs_summary,
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
            "step_agent_token_usage": step_usage,
            "task_agent_token_usage": dict(self.task_agent_token_usage),
        }
