"""
V6 LLM Policy for SurvivalWorld Benchmark

This module provides LLM policies supporting:
- Three model families: Gemini, Groq (LLaMA-70B), OpenAI (GPT-4o-mini)
- Seven scaffold conditions (6 base + tool support)
- Enhanced metrics logging (tokens, energy curves)
- Raw response logging for analysis

V6 Changes from V5:
- Added OpenAI GPT policy
- Added tool support scaffold
- Enhanced metrics (token counts, energy tracking)
- Unified interface across all model families
"""

import logging
import os
import time
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

from llm_society.rl.v6_prompts import (
    V6State,
    V6HistoryEntry,
    build_baseline_prompt,
    build_explicit_prompt,
    build_reasoning_prompt,
    build_tool_prompt,
    build_prompt_with_memory,
    parse_baseline_response,
    parse_reasoning_response,
    parse_tool_response,
    action_dict_to_idx,
    ACTION_NAMES_V6,
    get_scaffold_config,
)

logger = logging.getLogger(__name__)


@dataclass
class V6PolicyConfig:
    """Configuration for V6 LLM policy."""
    prompt_format: str = "baseline"  # "baseline", "explicit", "reasoning", "tool"
    memory_enabled: bool = False
    memory_window: int = 5
    tool_enabled: bool = False
    temperature: float = 0.0
    max_retries: int = 1
    fallback_action: str = "rest"


@dataclass
class V6StepLog:
    """Enhanced log entry for a single step."""
    step: int
    prompt: str
    raw_response: str
    parsed_action: Optional[Dict[str, Any]]
    thought_text: str  # For reasoning format
    action_text: str
    calc_text: str  # For tool format
    calc_results: List[str]  # For tool format
    parse_success: bool
    used_fallback: bool
    # V6 Enhanced metrics
    prompt_tokens: int
    response_tokens: int
    latency_ms: float
    energy_before: float
    energy_after: float  # Predicted based on action


# =============================================================================
# GEMINI POLICY
# =============================================================================

class V6GeminiPolicy:
    """
    V6 LLM Policy using Gemini.

    Supports all scaffold conditions with enhanced logging.
    """

    def __init__(
        self,
        config: V6PolicyConfig,
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
        self.backend = "gemini"

        # History for memory condition
        self.history: List[V6HistoryEntry] = []

        # Logging
        self.step_logs: List[V6StepLog] = []
        self.call_count = 0
        self.parse_failures = 0
        self.fallback_count = 0
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0

        logger.info(
            f"Initialized V6GeminiPolicy: model={model_name}, "
            f"format={config.prompt_format}, memory={config.memory_enabled}"
        )

    def reset(self):
        """Reset history and logs for new episode."""
        self.history = []
        self.step_logs = []

    def select_action(
        self,
        state: V6State,
        last_reward: float = 0.0,
        last_energy: Optional[float] = None,
        last_food: Optional[int] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select action given current state.

        Args:
            state: Current V6State
            last_reward: Reward from previous step (for history)
            last_energy: Energy after previous action (for history)
            last_food: Food after previous action (for history)

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
            elif self.config.prompt_format == "tool":
                prompt = build_tool_prompt(state)
            else:
                prompt = build_baseline_prompt(state)

        # Query LLM
        raw_response = ""
        parsed_action = None
        thought_text = ""
        action_text = ""
        calc_text = ""
        calc_results = []
        parse_success = False
        used_fallback = False
        prompt_tokens = 0
        response_tokens = 0
        latency_ms = 0.0

        start_time = time.time()

        for attempt in range(self.config.max_retries + 1):
            try:
                self.call_count += 1

                response = self.model.generate_content(
                    prompt,
                    generation_config=self.genai.GenerationConfig(
                        temperature=self.config.temperature,
                        max_output_tokens=300 if self.config.prompt_format in ["reasoning", "tool"] else 50,
                    )
                )

                raw_response = response.text.strip()

                # Estimate token counts (Gemini doesn't always provide)
                prompt_tokens = len(prompt.split()) * 1.3  # Rough estimate
                response_tokens = len(raw_response.split()) * 1.3

                # Parse based on format
                if self.config.prompt_format == "reasoning":
                    parsed_action, thought_text, action_text = parse_reasoning_response(raw_response)
                elif self.config.prompt_format == "tool":
                    parsed_action, calc_text, action_text, calc_results = parse_tool_response(raw_response)
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

        latency_ms = (time.time() - start_time) * 1000

        # Use fallback if parsing failed
        if parsed_action is None:
            parsed_action = {"type": "rest", "params": {}}
            used_fallback = True
            self.fallback_count += 1

        # Predict energy after action
        action_type = parsed_action.get("type", "rest")
        if action_type == "rest":
            energy_after = min(1.0, state.energy + 0.03)
        elif action_type == "gather_resources":
            energy_after = max(0.0, state.energy - 0.06)
        elif action_type == "move_to":
            energy_after = max(0.0, state.energy - 0.05)
        else:
            energy_after = max(0.0, state.energy - 0.02)

        # Log this step
        log_entry = V6StepLog(
            step=state.step,
            prompt=prompt,
            raw_response=raw_response,
            parsed_action=parsed_action,
            thought_text=thought_text,
            action_text=action_text,
            calc_text=calc_text,
            calc_results=calc_results,
            parse_success=parse_success,
            used_fallback=used_fallback,
            prompt_tokens=int(prompt_tokens),
            response_tokens=int(response_tokens),
            latency_ms=latency_ms,
            energy_before=state.energy,
            energy_after=energy_after,
        )
        self.step_logs.append(log_entry)
        self.total_prompt_tokens += int(prompt_tokens)
        self.total_response_tokens += int(response_tokens)

        # Update history for memory condition
        if self.config.memory_enabled:
            history_entry = V6HistoryEntry(
                step=state.step,
                observation=state,
                action=parsed_action.get("type", "rest"),
                reward=last_reward,
                distance_to_other=state.distance_to_other,
                energy_after=last_energy if last_energy is not None else state.energy,
                food_after=last_food if last_food is not None else state.food,
            )
            self.history.append(history_entry)

        action_idx = action_dict_to_idx(parsed_action)
        return action_idx, parsed_action

    def get_stats(self) -> Dict[str, Any]:
        """Return policy statistics."""
        return {
            "backend": self.backend,
            "model": self.model_name,
            "call_count": self.call_count,
            "parse_failures": self.parse_failures,
            "fallback_count": self.fallback_count,
            "parse_failure_rate": self.parse_failures / max(1, self.call_count),
            "fallback_rate": self.fallback_count / max(1, self.call_count),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_response_tokens": self.total_response_tokens,
        }

    def get_step_logs(self) -> List[Dict[str, Any]]:
        """Return step logs as list of dicts for serialization."""
        return [asdict(log) for log in self.step_logs]


# =============================================================================
# GROQ POLICY (LLaMA-70B)
# =============================================================================

class V6GroqPolicy:
    """
    V6 LLM Policy using Groq (LLaMA-70B).

    Supports all scaffold conditions with enhanced logging.
    """

    def __init__(
        self,
        config: V6PolicyConfig,
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
        self.backend = "groq"

        # History for memory condition
        self.history: List[V6HistoryEntry] = []

        # Logging
        self.step_logs: List[V6StepLog] = []
        self.call_count = 0
        self.parse_failures = 0
        self.fallback_count = 0
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0

        logger.info(
            f"Initialized V6GroqPolicy: model={model_name}, "
            f"format={config.prompt_format}, memory={config.memory_enabled}"
        )

    def reset(self):
        """Reset history and logs for new episode."""
        self.history = []
        self.step_logs = []

    def select_action(
        self,
        state: V6State,
        last_reward: float = 0.0,
        last_energy: Optional[float] = None,
        last_food: Optional[int] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Select action given current state."""
        # Build prompt
        if self.config.memory_enabled and self.history:
            prompt = build_prompt_with_memory(
                state, self.history, self.config.prompt_format, self.config.memory_window
            )
        else:
            if self.config.prompt_format == "baseline":
                prompt = build_baseline_prompt(state)
            elif self.config.prompt_format == "explicit":
                prompt = build_explicit_prompt(state)
            elif self.config.prompt_format == "reasoning":
                prompt = build_reasoning_prompt(state)
            elif self.config.prompt_format == "tool":
                prompt = build_tool_prompt(state)
            else:
                prompt = build_baseline_prompt(state)

        raw_response = ""
        parsed_action = None
        thought_text = ""
        action_text = ""
        calc_text = ""
        calc_results = []
        parse_success = False
        used_fallback = False
        prompt_tokens = 0
        response_tokens = 0
        latency_ms = 0.0

        start_time = time.time()

        for attempt in range(self.config.max_retries + 1):
            try:
                self.call_count += 1

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=300 if self.config.prompt_format in ["reasoning", "tool"] else 50,
                )

                raw_response = response.choices[0].message.content.strip()

                # Get token counts from response
                if hasattr(response, 'usage') and response.usage:
                    prompt_tokens = response.usage.prompt_tokens
                    response_tokens = response.usage.completion_tokens
                else:
                    prompt_tokens = len(prompt.split()) * 1.3
                    response_tokens = len(raw_response.split()) * 1.3

                # Parse based on format
                if self.config.prompt_format == "reasoning":
                    parsed_action, thought_text, action_text = parse_reasoning_response(raw_response)
                elif self.config.prompt_format == "tool":
                    parsed_action, calc_text, action_text, calc_results = parse_tool_response(raw_response)
                else:
                    parsed_action = parse_baseline_response(raw_response)
                    action_text = raw_response

                if parsed_action is not None:
                    parse_success = True
                    break
                else:
                    self.parse_failures += 1

            except Exception as e:
                logger.warning(f"Groq API error on attempt {attempt + 1}: {e}")
                time.sleep(1.0)

        latency_ms = (time.time() - start_time) * 1000

        if parsed_action is None:
            parsed_action = {"type": "rest", "params": {}}
            used_fallback = True
            self.fallback_count += 1

        # Predict energy after action
        action_type = parsed_action.get("type", "rest")
        if action_type == "rest":
            energy_after = min(1.0, state.energy + 0.03)
        elif action_type == "gather_resources":
            energy_after = max(0.0, state.energy - 0.06)
        elif action_type == "move_to":
            energy_after = max(0.0, state.energy - 0.05)
        else:
            energy_after = max(0.0, state.energy - 0.02)

        log_entry = V6StepLog(
            step=state.step,
            prompt=prompt,
            raw_response=raw_response,
            parsed_action=parsed_action,
            thought_text=thought_text,
            action_text=action_text,
            calc_text=calc_text,
            calc_results=calc_results,
            parse_success=parse_success,
            used_fallback=used_fallback,
            prompt_tokens=int(prompt_tokens),
            response_tokens=int(response_tokens),
            latency_ms=latency_ms,
            energy_before=state.energy,
            energy_after=energy_after,
        )
        self.step_logs.append(log_entry)
        self.total_prompt_tokens += int(prompt_tokens)
        self.total_response_tokens += int(response_tokens)

        if self.config.memory_enabled:
            history_entry = V6HistoryEntry(
                step=state.step,
                observation=state,
                action=parsed_action.get("type", "rest"),
                reward=last_reward,
                distance_to_other=state.distance_to_other,
                energy_after=last_energy if last_energy is not None else state.energy,
                food_after=last_food if last_food is not None else state.food,
            )
            self.history.append(history_entry)

        action_idx = action_dict_to_idx(parsed_action)
        return action_idx, parsed_action

    def get_stats(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model_name,
            "call_count": self.call_count,
            "parse_failures": self.parse_failures,
            "fallback_count": self.fallback_count,
            "parse_failure_rate": self.parse_failures / max(1, self.call_count),
            "fallback_rate": self.fallback_count / max(1, self.call_count),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_response_tokens": self.total_response_tokens,
        }

    def get_step_logs(self) -> List[Dict[str, Any]]:
        return [asdict(log) for log in self.step_logs]


# =============================================================================
# OPENAI POLICY (GPT-4o-mini)
# =============================================================================

class V6OpenAIPolicy:
    """
    V6 LLM Policy using OpenAI (GPT-4o-mini).

    Supports all scaffold conditions with enhanced logging.
    """

    def __init__(
        self,
        config: V6PolicyConfig,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o-mini",
    ):
        from openai import OpenAI

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.config = config
        self.model_name = model_name
        self.backend = "openai"

        # History for memory condition
        self.history: List[V6HistoryEntry] = []

        # Logging
        self.step_logs: List[V6StepLog] = []
        self.call_count = 0
        self.parse_failures = 0
        self.fallback_count = 0
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0

        logger.info(
            f"Initialized V6OpenAIPolicy: model={model_name}, "
            f"format={config.prompt_format}, memory={config.memory_enabled}"
        )

    def reset(self):
        """Reset history and logs for new episode."""
        self.history = []
        self.step_logs = []

    def select_action(
        self,
        state: V6State,
        last_reward: float = 0.0,
        last_energy: Optional[float] = None,
        last_food: Optional[int] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Select action given current state."""
        # Build prompt
        if self.config.memory_enabled and self.history:
            prompt = build_prompt_with_memory(
                state, self.history, self.config.prompt_format, self.config.memory_window
            )
        else:
            if self.config.prompt_format == "baseline":
                prompt = build_baseline_prompt(state)
            elif self.config.prompt_format == "explicit":
                prompt = build_explicit_prompt(state)
            elif self.config.prompt_format == "reasoning":
                prompt = build_reasoning_prompt(state)
            elif self.config.prompt_format == "tool":
                prompt = build_tool_prompt(state)
            else:
                prompt = build_baseline_prompt(state)

        raw_response = ""
        parsed_action = None
        thought_text = ""
        action_text = ""
        calc_text = ""
        calc_results = []
        parse_success = False
        used_fallback = False
        prompt_tokens = 0
        response_tokens = 0
        latency_ms = 0.0

        start_time = time.time()

        for attempt in range(self.config.max_retries + 1):
            try:
                self.call_count += 1

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=300 if self.config.prompt_format in ["reasoning", "tool"] else 50,
                )

                raw_response = response.choices[0].message.content.strip()

                # Get token counts from response
                if hasattr(response, 'usage') and response.usage:
                    prompt_tokens = response.usage.prompt_tokens
                    response_tokens = response.usage.completion_tokens
                else:
                    prompt_tokens = len(prompt.split()) * 1.3
                    response_tokens = len(raw_response.split()) * 1.3

                # Parse based on format
                if self.config.prompt_format == "reasoning":
                    parsed_action, thought_text, action_text = parse_reasoning_response(raw_response)
                elif self.config.prompt_format == "tool":
                    parsed_action, calc_text, action_text, calc_results = parse_tool_response(raw_response)
                else:
                    parsed_action = parse_baseline_response(raw_response)
                    action_text = raw_response

                if parsed_action is not None:
                    parse_success = True
                    break
                else:
                    self.parse_failures += 1

            except Exception as e:
                logger.warning(f"OpenAI API error on attempt {attempt + 1}: {e}")
                time.sleep(1.0)

        latency_ms = (time.time() - start_time) * 1000

        if parsed_action is None:
            parsed_action = {"type": "rest", "params": {}}
            used_fallback = True
            self.fallback_count += 1

        # Predict energy after action
        action_type = parsed_action.get("type", "rest")
        if action_type == "rest":
            energy_after = min(1.0, state.energy + 0.03)
        elif action_type == "gather_resources":
            energy_after = max(0.0, state.energy - 0.06)
        elif action_type == "move_to":
            energy_after = max(0.0, state.energy - 0.05)
        else:
            energy_after = max(0.0, state.energy - 0.02)

        log_entry = V6StepLog(
            step=state.step,
            prompt=prompt,
            raw_response=raw_response,
            parsed_action=parsed_action,
            thought_text=thought_text,
            action_text=action_text,
            calc_text=calc_text,
            calc_results=calc_results,
            parse_success=parse_success,
            used_fallback=used_fallback,
            prompt_tokens=int(prompt_tokens),
            response_tokens=int(response_tokens),
            latency_ms=latency_ms,
            energy_before=state.energy,
            energy_after=energy_after,
        )
        self.step_logs.append(log_entry)
        self.total_prompt_tokens += int(prompt_tokens)
        self.total_response_tokens += int(response_tokens)

        if self.config.memory_enabled:
            history_entry = V6HistoryEntry(
                step=state.step,
                observation=state,
                action=parsed_action.get("type", "rest"),
                reward=last_reward,
                distance_to_other=state.distance_to_other,
                energy_after=last_energy if last_energy is not None else state.energy,
                food_after=last_food if last_food is not None else state.food,
            )
            self.history.append(history_entry)

        action_idx = action_dict_to_idx(parsed_action)
        return action_idx, parsed_action

    def get_stats(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model_name,
            "call_count": self.call_count,
            "parse_failures": self.parse_failures,
            "fallback_count": self.fallback_count,
            "parse_failure_rate": self.parse_failures / max(1, self.call_count),
            "fallback_rate": self.fallback_count / max(1, self.call_count),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_response_tokens": self.total_response_tokens,
        }

    def get_step_logs(self) -> List[Dict[str, Any]]:
        return [asdict(log) for log in self.step_logs]


# =============================================================================
# POLICY FACTORY
# =============================================================================

def create_v6_policy(
    backend: str,  # "gemini", "groq", or "openai"
    scaffold: str,  # One of V6_SCAFFOLDS
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
):
    """
    Factory function to create V6 policy.

    Args:
        backend: "gemini", "groq", or "openai"
        scaffold: One of V6_SCAFFOLDS (e.g., "baseline_nomem", "reasoning_memory")
        api_key: API key (or uses environment variable)
        model_name: Override default model name
        temperature: Sampling temperature

    Returns:
        V6GeminiPolicy, V6GroqPolicy, or V6OpenAIPolicy instance
    """
    # Get scaffold configuration
    scaffold_config = get_scaffold_config(scaffold)

    config = V6PolicyConfig(
        prompt_format=scaffold_config["prompt_format"],
        memory_enabled=scaffold_config["memory_enabled"],
        tool_enabled=scaffold_config["tool_enabled"],
        memory_window=5,
        temperature=temperature,
    )

    # Default model names
    default_models = {
        "gemini": "gemini-2.0-flash",
        "groq": "llama-3.3-70b-versatile",
        "openai": "gpt-4o-mini",
    }

    model = model_name or default_models.get(backend)

    if backend == "gemini":
        return V6GeminiPolicy(config, api_key=api_key, model_name=model)
    elif backend == "groq":
        return V6GroqPolicy(config, api_key=api_key, model_name=model)
    elif backend == "openai":
        return V6OpenAIPolicy(config, api_key=api_key, model_name=model)
    else:
        raise ValueError(f"Unknown backend: {backend}. Valid: gemini, groq, openai")


# =============================================================================
# MODEL FAMILIES FOR V6
# =============================================================================

V6_MODEL_FAMILIES = {
    "gemini": {
        "backend": "gemini",
        "model": "gemini-2.0-flash",
        "env_key": "GOOGLE_API_KEY",
    },
    "groq": {
        "backend": "groq",
        "model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
    },
    "openai": {
        "backend": "openai",
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
}
