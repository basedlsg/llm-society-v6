"""
V5 Prompt Formats for LLM Policy Evaluation

This module implements the three prompt formats specified in V5:
(A) Baseline Prompt - Structurally identical to v4
(B) Explicit-Format Prompt - Explicit action syntax and consequences
(C) Reasoning-Enabled Prompt - Two-part output with THOUGHT and ACTION

All prompts contain identical environment information and action sets.
No prompt contains strategy hints, optimization suggestions, or goal-related advice.
Only the formatting differs.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class V5State:
    """State information for V5 prompts."""
    energy: float
    health: float
    food: int
    position_x: float
    position_y: float
    step: int
    max_steps: int
    distance_to_other: float
    other_agent_nearby: bool  # within social radius (3 units)
    other_position_x: Optional[float] = None
    other_position_y: Optional[float] = None


@dataclass
class V5HistoryEntry:
    """Single history entry for memory condition."""
    step: int
    observation: V5State
    action: str
    reward: float
    distance_to_other: float


# =============================================================================
# (A) BASELINE PROMPT - v4-equivalent
# =============================================================================

def build_baseline_prompt(state: V5State) -> str:
    """
    Build baseline prompt (v4-equivalent).

    Serves as control for comparability.
    No additional structure or coercion beyond required information.
    """
    if state.other_agent_nearby:
        presence_info = f"\n- Another agent is nearby ({state.distance_to_other:.1f} units away)"
    else:
        presence_info = f"\n- Another agent exists but is far ({state.distance_to_other:.1f} units away)"

    prompt = f"""You are an agent trying to survive in a 20x20 world. You share this world with one other agent.

Current state:
- Energy: {state.energy:.2f} (dies if reaches 0)
- Health: {state.health:.2f} (dies if reaches 0)
- Food: {state.food} units (1 consumed every 10 steps; if 0, health drops)
- Step: {state.step} of {state.max_steps}{presence_info}

Actions available:
- rest: recovers energy
- gather_resources: costs energy, gives food
- move_to X Y: costs energy, moves toward position (X, Y)

What is your action? Reply with exactly one action."""

    return prompt


# =============================================================================
# (B) EXPLICIT-FORMAT PROMPT
# =============================================================================

def build_explicit_prompt(state: V5State) -> str:
    """
    Build explicit-format prompt.

    Contains:
    - Explicit action syntax
    - Explicit valid-response rules
    - Explicit descriptions of action costs and consequences

    No instructions about strategy or interpretation.
    Format-only modification to reduce ambiguity.
    """
    if state.other_agent_nearby:
        presence_info = f"NEARBY ({state.distance_to_other:.1f} units)"
        other_pos = f"Other agent position: ({state.other_position_x:.1f}, {state.other_position_y:.1f})"
    else:
        presence_info = f"FAR ({state.distance_to_other:.1f} units)"
        other_pos = f"Other agent position: ({state.other_position_x:.1f}, {state.other_position_y:.1f})"

    prompt = f"""SURVIVAL WORLD STATE
====================
Your position: ({state.position_x:.1f}, {state.position_y:.1f})
Energy: {state.energy:.2f} / 1.00
Health: {state.health:.2f} / 1.00
Food: {state.food} units
Step: {state.step} / {state.max_steps}
Other agent: {presence_info}
{other_pos}

WORLD RULES
===========
- World size: 20x20 (coordinates 0-20)
- Energy decays -0.02 per step automatically
- Food consumed: 1 unit every 10 steps
- If food = 0: health decreases -0.1 per food tick
- Death occurs if energy <= 0 OR health <= 0

COOPERATIVE MECHANIC
====================
- If both agents gather within 2 units: each gets +5 food
- If only one gathers or too far apart: normal +1-3 food

VALID ACTIONS (choose exactly one)
==================================
1. rest
   - Syntax: rest
   - Effect: +0.03 net energy (+0.05 gain, -0.02 decay)

2. gather_resources
   - Syntax: gather_resources
   - Effect: -0.06 net energy, +1-3 food (or +5 if cooperative)

3. move_to X Y
   - Syntax: move_to X Y (where X and Y are numbers 0-20)
   - Effect: -0.05 net energy, moves 1 unit toward (X, Y)
   - Example: move_to 10.0 15.0

RESPONSE FORMAT
===============
Reply with ONLY the action. No explanation. No punctuation.
Valid responses: "rest" or "gather_resources" or "move_to X Y"

YOUR ACTION:"""

    return prompt


# =============================================================================
# (C) REASONING-ENABLED PROMPT (Two-Part Output)
# =============================================================================

def build_reasoning_prompt(state: V5State) -> str:
    """
    Build reasoning-enabled prompt with two-part output.

    First part: constrained natural-language reasoning section
    Second part: strict action token

    Parser only reads <ACTION>...</ACTION> token.
    """
    if state.other_agent_nearby:
        presence_info = f"NEARBY ({state.distance_to_other:.1f} units)"
        other_pos = f"Position: ({state.other_position_x:.1f}, {state.other_position_y:.1f})"
    else:
        presence_info = f"FAR ({state.distance_to_other:.1f} units)"
        other_pos = f"Position: ({state.other_position_x:.1f}, {state.other_position_y:.1f})"

    prompt = f"""You are an agent in a survival world. Observe your state and choose an action.

CURRENT STATE
=============
Your position: ({state.position_x:.1f}, {state.position_y:.1f})
Energy: {state.energy:.2f}
Health: {state.health:.2f}
Food: {state.food}
Step: {state.step} / {state.max_steps}
Other agent: {presence_info}
  {other_pos}

MECHANICS
=========
- Energy decays -0.02/step. Death at energy=0 or health=0.
- Food consumed 1/10 steps. No food = health drops.
- Cooperative gather: both agents gather within 2 units = +5 food each.

ACTIONS
=======
- rest: +0.03 net energy
- gather_resources: -0.06 net energy, +1-3 food (or +5 cooperative)
- move_to X Y: -0.05 net energy, move toward (X,Y)

RESPONSE FORMAT
===============
You MUST respond in this exact format:

<THOUGHT>
(Your reasoning here - what you observe and why you choose your action)
</THOUGHT>
<ACTION>
(exactly one of: rest | gather_resources | move_to X Y)
</ACTION>

Respond now:"""

    return prompt


# =============================================================================
# MEMORY CONDITION - History Block Builder
# =============================================================================

def build_history_block(history: List[V5HistoryEntry], max_entries: int = 5) -> str:
    """
    Build history block for memory condition.

    Contains past N steps including:
    - Prior observation
    - Prior action
    - Prior reward
    - Prior distance to partner

    Included verbatim in prompt. No labels, summaries, or interpretation.
    """
    if not history:
        return "HISTORY: (no prior steps)"

    # Take last N entries
    recent = history[-max_entries:]

    lines = ["RECENT HISTORY (last {} steps)".format(len(recent))]
    lines.append("=" * 40)

    for entry in recent:
        lines.append(f"Step {entry.step}:")
        lines.append(f"  State: energy={entry.observation.energy:.2f}, health={entry.observation.health:.2f}, food={entry.observation.food}")
        lines.append(f"  Position: ({entry.observation.position_x:.1f}, {entry.observation.position_y:.1f})")
        lines.append(f"  Distance to other: {entry.distance_to_other:.1f}")
        lines.append(f"  Action taken: {entry.action}")
        lines.append(f"  Reward received: {entry.reward:.3f}")
        lines.append("")

    return "\n".join(lines)


def build_prompt_with_memory(
    state: V5State,
    history: List[V5HistoryEntry],
    prompt_type: str = "baseline",
    max_history: int = 5
) -> str:
    """
    Build prompt with memory (history) prepended.

    Args:
        state: Current state
        history: List of past observations/actions/rewards
        prompt_type: One of "baseline", "explicit", "reasoning"
        max_history: Maximum number of history entries (default 5)

    Returns:
        Complete prompt with history block prepended
    """
    history_block = build_history_block(history, max_history)

    if prompt_type == "baseline":
        base_prompt = build_baseline_prompt(state)
    elif prompt_type == "explicit":
        base_prompt = build_explicit_prompt(state)
    elif prompt_type == "reasoning":
        base_prompt = build_reasoning_prompt(state)
    else:
        raise ValueError(f"Unknown prompt_type: {prompt_type}")

    # Prepend history to prompt
    return f"{history_block}\n\n{base_prompt}"


# =============================================================================
# PARSER FUNCTIONS
# =============================================================================

def parse_baseline_response(response: str) -> Optional[Dict[str, Any]]:
    """
    Parse response from baseline or explicit prompt.

    Returns action dict or None if parsing fails.
    """
    response = response.strip().lower()

    # Direct matches
    if response == "rest":
        return {"type": "rest", "params": {}}

    if response == "gather_resources" or response == "gather":
        return {"type": "gather_resources", "params": {}}

    # Move with coordinates
    if response.startswith("move_to") or response.startswith("move "):
        parts = response.replace("move_to", "").replace("move", "").strip().split()
        if len(parts) >= 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                # Clamp to world bounds
                x = max(0.0, min(20.0, x))
                y = max(0.0, min(20.0, y))
                return {"type": "move_to", "params": {"x": x, "y": y}}
            except ValueError:
                pass

    # Fallback checks
    if "rest" in response:
        return {"type": "rest", "params": {}}
    if "gather" in response:
        return {"type": "gather_resources", "params": {}}
    if "move" in response:
        # Try to extract any numbers
        import re
        numbers = re.findall(r"[\d.]+", response)
        if len(numbers) >= 2:
            try:
                x = float(numbers[0])
                y = float(numbers[1])
                x = max(0.0, min(20.0, x))
                y = max(0.0, min(20.0, y))
                return {"type": "move_to", "params": {"x": x, "y": y}}
            except ValueError:
                pass

    return None


def parse_reasoning_response(response: str) -> tuple[Optional[Dict[str, Any]], str, str]:
    """
    Parse response from reasoning prompt.

    Extracts THOUGHT and ACTION sections.
    Parser only uses ACTION for the actual action.

    Returns:
        (action_dict, thought_text, action_text)
    """
    import re

    # Extract THOUGHT section
    thought_match = re.search(r"<THOUGHT>(.*?)</THOUGHT>", response, re.DOTALL | re.IGNORECASE)
    thought_text = thought_match.group(1).strip() if thought_match else ""

    # Extract ACTION section
    action_match = re.search(r"<ACTION>(.*?)</ACTION>", response, re.DOTALL | re.IGNORECASE)
    action_text = action_match.group(1).strip() if action_match else response.strip()

    # Parse the action
    action_dict = parse_baseline_response(action_text)

    return action_dict, thought_text, action_text


# =============================================================================
# V5 ACTION NAMES
# =============================================================================

ACTION_NAMES_V5 = ["rest", "gather_resources", "move_to"]
ACTION_TO_IDX_V5 = {name: idx for idx, name in enumerate(ACTION_NAMES_V5)}


def action_dict_to_idx(action: Optional[Dict[str, Any]]) -> int:
    """Convert action dict to index. Returns 0 (rest) if None."""
    if action is None:
        return 0
    action_type = action.get("type", "rest")
    return ACTION_TO_IDX_V5.get(action_type, 0)
