"""
Gemini Policy for SurvivalWorld Benchmark

This module provides a policy that uses Google's Gemini model to select actions
in the SurvivalWorld environment. Used for comparing LLM decision-making against
heuristic and learned (RL) policies.

Usage:
    from llm_society.rl.gemini_policy import GeminiPolicy

    policy = GeminiPolicy(api_key="...", temperature=0.0)
    action_idx, _ = policy.select_action(state)
"""

import logging
import os
import time
from typing import Tuple, Optional, List

import google.generativeai as genai

logger = logging.getLogger(__name__)

# Action mapping
ACTION_NAMES = ["rest", "gather_resources", "move_to", "talk_to"]
ACTION_TO_IDX = {name: idx for idx, name in enumerate(ACTION_NAMES)}


def build_prompt(state: List[float], max_steps: int = 200) -> str:
    """
    Build the prompt for Gemini given the current state.

    State indices:
        0: energy (0-1)
        1: health (0-1)
        2: food_normalized (food/10)
        3: position_x_norm
        4: position_y_norm
        5: step_fraction (step/200)
        6: connections_norm (unused in v1)
        7: nearby_norm (unused in v1)
    """
    energy = state[0]
    health = state[1]
    food = int(state[2] * 10)  # denormalize
    step = int(state[5] * max_steps)

    prompt = f"""You are an agent trying to survive as long as possible in a simple world by managing your energy, food, and health.

Current state:
- Energy: {energy:.2f} (dies if reaches 0)
- Health: {health:.2f} (dies if reaches 0)
- Food: {food} units (1 is consumed every 10 steps; if 0, health drops)
- Step: {step} of {max_steps}

Actions available:
- rest: increases your energy slightly but does not give food
- gather_resources: gives +1 to +3 food, but costs energy
- move_to: changes your position, costs energy, does not give food
- talk_to: no effect (no other agents present)

Your goal is to survive all {max_steps} steps by keeping your health and energy above 0 and not running out of food.

What is your action? Reply with exactly one word: rest, gather_resources, move_to, or talk_to."""

    return prompt


def build_prompt_v2(state: List[float], other_agent_nearby: bool, max_steps: int = 200) -> str:
    """
    Build the prompt for v2 (2-agent world).

    Same as v1 but acknowledges presence of another agent.
    """
    energy = state[0]
    health = state[1]
    food = int(state[2] * 10)  # denormalize
    step = int(state[5] * max_steps)
    nearby_count = int(state[7] * 5)  # denormalize nearby_norm

    # Presence information
    presence_info = ""
    if other_agent_nearby or nearby_count > 0:
        presence_info = "\n- Another agent is nearby (within detection range)"
    else:
        presence_info = "\n- No other agents nearby"

    prompt = f"""You are an agent trying to survive as long as possible in a simple world by managing your energy, food, and health.

Current state:
- Energy: {energy:.2f} (dies if reaches 0)
- Health: {health:.2f} (dies if reaches 0)
- Food: {food} units (1 is consumed every 10 steps; if 0, health drops)
- Step: {step} of {max_steps}{presence_info}

Actions available:
- rest: increases your energy slightly but does not give food
- gather_resources: gives +1 to +3 food, but costs energy
- move_to: changes your position, costs energy, does not give food
- talk_to: interact with nearby agent (costs energy, no survival benefit)

Your goal is to survive all {max_steps} steps by keeping your health and energy above 0 and not running out of food.

What is your action? Reply with exactly one word: rest, gather_resources, move_to, or talk_to."""

    return prompt


def parse_action(response: str) -> Optional[int]:
    """
    Parse Gemini's response to extract action index.

    Returns action index (0-3) or None if parsing fails.
    """
    response = response.strip().lower()

    # Direct match
    if response in ACTION_TO_IDX:
        return ACTION_TO_IDX[response]

    # Check if response contains exactly one action name
    found_actions = [name for name in ACTION_NAMES if name in response]
    if len(found_actions) == 1:
        return ACTION_TO_IDX[found_actions[0]]

    # Check for common variations
    if "gather" in response:
        return ACTION_TO_IDX["gather_resources"]
    if "move" in response:
        return ACTION_TO_IDX["move_to"]
    if "talk" in response:
        return ACTION_TO_IDX["talk_to"]

    return None


class GeminiPolicy:
    """
    Policy that uses Gemini to select actions in SurvivalWorld.

    Attributes:
        model: The Gemini model instance
        temperature: Sampling temperature (0 = deterministic)
        max_retries: Number of retries on parse failure
        fallback_action: Action to use if all retries fail
        call_count: Number of API calls made
        parse_failures: Number of parse failures encountered
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.0,
        max_retries: int = 1,
        fallback_action: int = 0,  # rest
        v2_mode: bool = False,
    ):
        """
        Initialize Gemini policy.

        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            model_name: Gemini model to use
            temperature: Sampling temperature (0 = deterministic)
            max_retries: Number of retries on parse failure
            fallback_action: Action index to use if all parsing fails
            v2_mode: If True, use v2 prompt with presence information
        """
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature
        self.max_retries = max_retries
        self.fallback_action = fallback_action
        self.v2_mode = v2_mode

        # Tracking
        self.call_count = 0
        self.parse_failures = 0
        self.fallback_count = 0

        # For compatibility with trainer.py interface
        self.saved_log_probs = []
        self.rewards = []

        logger.info(f"Initialized GeminiPolicy with model={model_name}, temp={temperature}, v2={v2_mode}")

    def select_action(self, state, other_agent_nearby: bool = False) -> Tuple[int, float]:
        """
        Select action given state vector.

        Args:
            state: numpy array or list of 8 state features
            other_agent_nearby: Whether another agent is nearby (for v2 mode)

        Returns:
            (action_index, log_prob) - log_prob is always 0.0 for LLM policy
        """
        # Convert numpy array to list if needed
        if hasattr(state, 'tolist'):
            state = state.tolist()

        if self.v2_mode:
            prompt = build_prompt_v2(state, other_agent_nearby)
        else:
            prompt = build_prompt(state)

        for attempt in range(self.max_retries + 1):
            try:
                self.call_count += 1

                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=10,
                    )
                )

                response_text = response.text.strip()
                action_idx = parse_action(response_text)

                if action_idx is not None:
                    return action_idx, 0.0

                # Parse failed
                self.parse_failures += 1
                logger.debug(f"Parse failed on attempt {attempt + 1}: '{response_text}'")

                if attempt < self.max_retries:
                    # Retry with stricter prompt
                    prompt = "You must answer with only one of: rest, gather_resources, move_to, talk_to. No other words."

            except Exception as e:
                logger.warning(f"Gemini API error on attempt {attempt + 1}: {e}")
                time.sleep(0.5)  # Brief pause before retry

        # All attempts failed, use fallback
        self.fallback_count += 1
        logger.debug(f"Using fallback action after {self.max_retries + 1} attempts")
        return self.fallback_action, 0.0

    def get_stats(self) -> dict:
        """Return policy statistics."""
        return {
            "call_count": self.call_count,
            "parse_failures": self.parse_failures,
            "fallback_count": self.fallback_count,
            "parse_failure_rate": self.parse_failures / max(1, self.call_count),
            "fallback_rate": self.fallback_count / max(1, self.call_count),
        }

    def reset_stats(self):
        """Reset tracking statistics."""
        self.call_count = 0
        self.parse_failures = 0
        self.fallback_count = 0


def test_policy():
    """Quick test of the Gemini policy."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not set, skipping test")
        return

    policy = GeminiPolicy(temperature=0.0)

    # Test with a few different states
    test_states = [
        [1.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0],   # Fresh start
        [0.2, 1.0, 0.5, 0.5, 0.5, 0.25, 0.0, 0.0],  # Low energy
        [1.0, 1.0, 0.1, 0.5, 0.5, 0.5, 0.0, 0.0],   # Low food
        [0.3, 0.5, 0.0, 0.5, 0.5, 0.75, 0.0, 0.0],  # Critical: low health, no food
    ]

    print("Testing GeminiPolicy...")
    for i, state in enumerate(test_states):
        action_idx, _ = policy.select_action(state)
        action_name = ACTION_NAMES[action_idx]
        print(f"  State {i + 1}: energy={state[0]:.1f}, food={int(state[2]*10)}, health={state[1]:.1f} -> {action_name}")

    print(f"\nStats: {policy.get_stats()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_policy()
