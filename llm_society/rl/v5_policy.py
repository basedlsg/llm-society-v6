"""
V5 LLM Policy for SurvivalWorld Benchmark

This module provides LLM policies with:
- Three prompt formats (baseline, explicit, reasoning)
- Optional memory condition (sliding window history)
- Raw response logging for analysis

Supports both Gemini and Groq backends.
"""

import logging
import os
import time
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field

from llm_society.rl.v5_prompts import (
    V5State,
    V5HistoryEntry,
    build_baseline_prompt,
    build_explicit_prompt,
    build_reasoning_prompt,
    build_prompt_with_memory,
    parse_baseline_response,
    parse_reasoning_response,
    action_dict_to_idx,
    ACTION_NAMES_V5,
)

logger = logging.getLogger(__name__)


@dataclass
class V5PolicyConfig:
    """Configuration for V5 LLM policy."""
    prompt_format: str = "baseline"  # "baseline", "explicit", or "reasoning"
    memory_enabled: bool = False
    memory_window: int = 5
    temperature: float = 0.0
    max_retries: int = 1
    fallback_action: int = 0  # rest


@dataclass
class V5StepLog:
    """Log entry for a single step."""
    step: int
    state: V5State
    prompt: str
    raw_response: str
    parsed_action: Optional[Dict[str, Any]]
    thought_text: str  # Only for reasoning format
    action_text: str
    parse_success: bool
    used_fallback: bool


class V5GeminiPolicy:
    """
    V5 LLM Policy using Gemini.

    Supports all three prompt formats and optional memory.
    Logs raw responses for analysis.
    """

    def __init__(
        self,
        config: V5PolicyConfig,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
    ):
        import google.generativeai as genai

        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.genai = genai
        self.config = config
        self.model_name = model_name

        # History for memory condition
        self.history: List[V5HistoryEntry] = []

        # Logging
        self.step_logs: List[V5StepLog] = []
        self.call_count = 0
        self.parse_failures = 0
        self.fallback_count = 0

        logger.info(
            f"Initialized V5GeminiPolicy: model={model_name}, "
            f"format={config.prompt_format}, memory={config.memory_enabled}"
        )

    def reset(self):
        """Reset history and logs for new episode."""
        self.history = []
        self.step_logs = []

    def select_action(
        self,
        state: V5State,
        last_reward: float = 0.0,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select action given current state.

        Args:
            state: Current V5State
            last_reward: Reward from previous step (for history)

        Returns:
            (action_index, action_dict)
        """
        # Build prompt based on format and memory setting
        if self.config.memory_enabled and self.history:
            prompt = build_prompt_with_memory(
                state,
                self.history,
                self.config.prompt_format,
                self.config.memory_window,
            )
        else:
            if self.config.prompt_format == "baseline":
                prompt = build_baseline_prompt(state)
            elif self.config.prompt_format == "explicit":
                prompt = build_explicit_prompt(state)
            elif self.config.prompt_format == "reasoning":
                prompt = build_reasoning_prompt(state)
            else:
                prompt = build_baseline_prompt(state)

        # Query LLM
        raw_response = ""
        parsed_action = None
        thought_text = ""
        action_text = ""
        parse_success = False
        used_fallback = False

        for attempt in range(self.config.max_retries + 1):
            try:
                self.call_count += 1

                response = self.model.generate_content(
                    prompt,
                    generation_config=self.genai.GenerationConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=200 if self.config.prompt_format == "reasoning" else 50,
                    )
                )

                raw_response = response.text.strip()

                # Parse based on format
                if self.config.prompt_format == "reasoning":
                    parsed_action, thought_text, action_text = parse_reasoning_response(raw_response)
                else:
                    parsed_action = parse_baseline_response(raw_response)
                    action_text = raw_response

                if parsed_action is not None:
                    parse_success = True
                    break
                else:
                    self.parse_failures += 1
                    logger.debug(f"Parse failed on attempt {attempt + 1}: '{raw_response}'")

            except Exception as e:
                logger.warning(f"Gemini API error on attempt {attempt + 1}: {e}")
                time.sleep(0.5)

        # Use fallback if parsing failed
        if parsed_action is None:
            parsed_action = {"type": "rest", "params": {}}
            used_fallback = True
            self.fallback_count += 1

        # Log this step
        log_entry = V5StepLog(
            step=state.step,
            state=state,
            prompt=prompt,
            raw_response=raw_response,
            parsed_action=parsed_action,
            thought_text=thought_text,
            action_text=action_text,
            parse_success=parse_success,
            used_fallback=used_fallback,
        )
        self.step_logs.append(log_entry)

        # Update history for memory condition
        if self.config.memory_enabled:
            history_entry = V5HistoryEntry(
                step=state.step,
                observation=state,
                action=parsed_action.get("type", "rest"),
                reward=last_reward,
                distance_to_other=state.distance_to_other,
            )
            self.history.append(history_entry)

        action_idx = action_dict_to_idx(parsed_action)
        return action_idx, parsed_action

    def get_stats(self) -> Dict[str, Any]:
        """Return policy statistics."""
        return {
            "call_count": self.call_count,
            "parse_failures": self.parse_failures,
            "fallback_count": self.fallback_count,
            "parse_failure_rate": self.parse_failures / max(1, self.call_count),
            "fallback_rate": self.fallback_count / max(1, self.call_count),
        }

    def get_step_logs(self) -> List[Dict[str, Any]]:
        """Return step logs as list of dicts for serialization."""
        logs = []
        for log in self.step_logs:
            logs.append({
                "step": log.step,
                "prompt": log.prompt,
                "raw_response": log.raw_response,
                "parsed_action": log.parsed_action,
                "thought_text": log.thought_text,
                "action_text": log.action_text,
                "parse_success": log.parse_success,
                "used_fallback": log.used_fallback,
            })
        return logs


class V5GroqPolicy:
    """
    V5 LLM Policy using Groq.

    Supports all three prompt formats and optional memory.
    Logs raw responses for analysis.
    """

    def __init__(
        self,
        config: V5PolicyConfig,
        api_key: Optional[str] = None,
        model_name: str = "llama-3.3-70b-versatile",
    ):
        from groq import Groq

        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

        self.client = Groq(api_key=api_key)
        self.config = config
        self.model_name = model_name

        # History for memory condition
        self.history: List[V5HistoryEntry] = []

        # Logging
        self.step_logs: List[V5StepLog] = []
        self.call_count = 0
        self.parse_failures = 0
        self.fallback_count = 0

        logger.info(
            f"Initialized V5GroqPolicy: model={model_name}, "
            f"format={config.prompt_format}, memory={config.memory_enabled}"
        )

    def reset(self):
        """Reset history and logs for new episode."""
        self.history = []
        self.step_logs = []

    def select_action(
        self,
        state: V5State,
        last_reward: float = 0.0,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select action given current state.

        Args:
            state: Current V5State
            last_reward: Reward from previous step (for history)

        Returns:
            (action_index, action_dict)
        """
        # Build prompt based on format and memory setting
        if self.config.memory_enabled and self.history:
            prompt = build_prompt_with_memory(
                state,
                self.history,
                self.config.prompt_format,
                self.config.memory_window,
            )
        else:
            if self.config.prompt_format == "baseline":
                prompt = build_baseline_prompt(state)
            elif self.config.prompt_format == "explicit":
                prompt = build_explicit_prompt(state)
            elif self.config.prompt_format == "reasoning":
                prompt = build_reasoning_prompt(state)
            else:
                prompt = build_baseline_prompt(state)

        # Query LLM
        raw_response = ""
        parsed_action = None
        thought_text = ""
        action_text = ""
        parse_success = False
        used_fallback = False

        for attempt in range(self.config.max_retries + 1):
            try:
                self.call_count += 1

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=200 if self.config.prompt_format == "reasoning" else 50,
                )

                raw_response = response.choices[0].message.content.strip()

                # Parse based on format
                if self.config.prompt_format == "reasoning":
                    parsed_action, thought_text, action_text = parse_reasoning_response(raw_response)
                else:
                    parsed_action = parse_baseline_response(raw_response)
                    action_text = raw_response

                if parsed_action is not None:
                    parse_success = True
                    break
                else:
                    self.parse_failures += 1
                    logger.debug(f"Parse failed on attempt {attempt + 1}: '{raw_response}'")

            except Exception as e:
                logger.warning(f"Groq API error on attempt {attempt + 1}: {e}")
                time.sleep(1.0)  # Longer pause for rate limits

        # Use fallback if parsing failed
        if parsed_action is None:
            parsed_action = {"type": "rest", "params": {}}
            used_fallback = True
            self.fallback_count += 1

        # Log this step
        log_entry = V5StepLog(
            step=state.step,
            state=state,
            prompt=prompt,
            raw_response=raw_response,
            parsed_action=parsed_action,
            thought_text=thought_text,
            action_text=action_text,
            parse_success=parse_success,
            used_fallback=used_fallback,
        )
        self.step_logs.append(log_entry)

        # Update history for memory condition
        if self.config.memory_enabled:
            history_entry = V5HistoryEntry(
                step=state.step,
                observation=state,
                action=parsed_action.get("type", "rest"),
                reward=last_reward,
                distance_to_other=state.distance_to_other,
            )
            self.history.append(history_entry)

        action_idx = action_dict_to_idx(parsed_action)
        return action_idx, parsed_action

    def get_stats(self) -> Dict[str, Any]:
        """Return policy statistics."""
        return {
            "call_count": self.call_count,
            "parse_failures": self.parse_failures,
            "fallback_count": self.fallback_count,
            "parse_failure_rate": self.parse_failures / max(1, self.call_count),
            "fallback_rate": self.fallback_count / max(1, self.call_count),
        }

    def get_step_logs(self) -> List[Dict[str, Any]]:
        """Return step logs as list of dicts for serialization."""
        logs = []
        for log in self.step_logs:
            logs.append({
                "step": log.step,
                "prompt": log.prompt,
                "raw_response": log.raw_response,
                "parsed_action": log.parsed_action,
                "thought_text": log.thought_text,
                "action_text": log.action_text,
                "parse_success": log.parse_success,
                "used_fallback": log.used_fallback,
            })
        return logs


# =============================================================================
# POLICY FACTORY
# =============================================================================

def create_v5_policy(
    backend: str,  # "gemini" or "groq"
    prompt_format: str,  # "baseline", "explicit", "reasoning"
    memory_enabled: bool = False,
    memory_window: int = 5,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
):
    """
    Factory function to create V5 policy.

    Args:
        backend: "gemini" or "groq"
        prompt_format: "baseline", "explicit", or "reasoning"
        memory_enabled: Whether to enable sliding window history
        memory_window: Number of past steps to include
        temperature: Sampling temperature
        api_key: API key (or uses environment variable)

    Returns:
        V5GeminiPolicy or V5GroqPolicy instance
    """
    config = V5PolicyConfig(
        prompt_format=prompt_format,
        memory_enabled=memory_enabled,
        memory_window=memory_window,
        temperature=temperature,
    )

    if backend == "gemini":
        return V5GeminiPolicy(config, api_key=api_key)
    elif backend == "groq":
        return V5GroqPolicy(config, api_key=api_key)
    else:
        raise ValueError(f"Unknown backend: {backend}")
