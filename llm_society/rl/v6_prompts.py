"""
V6 Prompt Formats for LLM Policy Evaluation

This module implements the seven scaffold conditions specified in V6:
(A) Baseline Prompt - Simple survival prompt (no memory, no tools)
(B) Baseline + Memory - With 5-step sliding window history
(C) Explicit Format - Explicit action syntax and reward rules
(D) Explicit + Memory - Explicit format with history
(E) Reasoning (CoT) - THOUGHT/ACTION two-part format
(F) Reasoning + Memory - CoT with history
(G) Tool Support - Explicit format with calculator tool access

V6 Changes from V5:
- talk_to action disabled
- Tool support scaffold added
- Consistent formatting across all model families
- Enhanced mechanics description (energy curves)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class V6State:
    """State information for V6 prompts."""
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
class V6HistoryEntry:
    """Single history entry for memory condition."""
    step: int
    observation: V6State
    action: str
    reward: float
    distance_to_other: float
    energy_after: float  # V6: track energy after action
    food_after: int  # V6: track food after action


# =============================================================================
# V6 ACTION SPACE (talk_to disabled)
# =============================================================================

ACTION_NAMES_V6 = ["rest", "gather_resources", "move_to"]
ACTION_TO_IDX_V6 = {name: idx for idx, name in enumerate(ACTION_NAMES_V6)}


def action_dict_to_idx(action: Optional[Dict[str, Any]]) -> int:
    """Convert action dict to index. Returns 0 (rest) if None."""
    if action is None:
        return 0
    action_type = action.get("type", "rest")
    return ACTION_TO_IDX_V6.get(action_type, 0)


# =============================================================================
# (A) BASELINE PROMPT - Simple survival prompt
# =============================================================================

def build_baseline_prompt(state: V6State) -> str:
    """
    Build baseline prompt (v4-equivalent, v6 physics).

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
- rest: recovers energy (+0.03 net)
- gather_resources: costs energy (-0.06 net), gives food (+1-3)
- move_to X Y: costs energy (-0.05 net), moves toward position (X, Y)

What is your action? Reply with exactly one action."""

    return prompt


# =============================================================================
# (B) EXPLICIT-FORMAT PROMPT
# =============================================================================

def build_explicit_prompt(state: V6State) -> str:
    """
    Build explicit-format prompt.

    Contains:
    - Explicit action syntax
    - Explicit valid-response rules
    - Explicit descriptions of action costs and consequences
    - Energy curve warnings

    No instructions about strategy or interpretation.
    Format-only modification to reduce ambiguity.
    """
    if state.other_agent_nearby:
        presence_info = f"NEARBY ({state.distance_to_other:.1f} units)"
        other_pos = f"Other agent position: ({state.other_position_x:.1f}, {state.other_position_y:.1f})"
    else:
        presence_info = f"FAR ({state.distance_to_other:.1f} units)"
        other_pos = f"Other agent position: ({state.other_position_x:.1f}, {state.other_position_y:.1f})"

    # Energy warning
    if state.energy < 0.2:
        energy_warning = "WARNING: CRITICAL ENERGY - rest immediately or die!"
    elif state.energy < 0.4:
        energy_warning = "CAUTION: Low energy - consider resting soon."
    else:
        energy_warning = ""

    prompt = f"""SURVIVAL WORLD STATE
====================
Your position: ({state.position_x:.1f}, {state.position_y:.1f})
Energy: {state.energy:.2f} / 1.00
Health: {state.health:.2f} / 1.00
Food: {state.food} units
Step: {state.step} / {state.max_steps}
Other agent: {presence_info}
{other_pos}
{energy_warning}

WORLD RULES
===========
- World size: 20x20 (coordinates 0-20)
- Energy decays -0.02 per step automatically
- Food consumed: 1 unit every 10 steps
- If food = 0 at consumption: health decreases -0.10
- Death occurs if energy <= 0 OR health <= 0

ENERGY MATH (CRITICAL)
======================
- rest: +0.05 gain - 0.02 decay = +0.03 NET
- gather_resources: -0.04 cost - 0.02 decay = -0.06 NET
- move_to: -0.03 cost - 0.02 decay = -0.05 NET

COOPERATIVE MECHANIC
====================
- If both agents gather within 2 units: each gets +5 food
- If only one gathers or too far apart: normal +1-3 food

VALID ACTIONS (choose exactly one)
==================================
1. rest
   - Syntax: rest
   - Effect: +0.03 net energy

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
# (C) REASONING-ENABLED PROMPT (CoT)
# =============================================================================

def build_reasoning_prompt(state: V6State) -> str:
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
- Energy decays -0.02/step automatically
- Death at energy<=0 or health<=0
- Food consumed 1/10 steps. No food = health drops -0.10
- rest: +0.03 net energy
- gather_resources: -0.06 net energy, +1-3 food
- move_to X Y: -0.05 net energy
- Cooperative gather: both agents gather within 2 units = +5 food each

RESPONSE FORMAT
===============
You MUST respond in this exact format:

<THOUGHT>
(Your reasoning: what you observe, calculate energy needs, decide action)
</THOUGHT>
<ACTION>
(exactly one of: rest | gather_resources | move_to X Y)
</ACTION>

Respond now:"""

    return prompt


# =============================================================================
# (D) TOOL SUPPORT PROMPT
# =============================================================================

def build_tool_prompt(state: V6State) -> str:
    """
    Build tool-support prompt with calculator access.

    LLM can request calculations before deciding action.
    Parser handles <CALC>...</CALC> and <ACTION>...</ACTION> sections.
    """
    if state.other_agent_nearby:
        presence_info = f"NEARBY ({state.distance_to_other:.1f} units)"
        other_pos = f"Position: ({state.other_position_x:.1f}, {state.other_position_y:.1f})"
    else:
        presence_info = f"FAR ({state.distance_to_other:.1f} units)"
        other_pos = f"Position: ({state.other_position_x:.1f}, {state.other_position_y:.1f})"

    # Pre-compute useful values for the tool context
    steps_until_food_consume = 10 - (state.step % 10)
    energy_if_rest = min(1.0, state.energy + 0.03)
    energy_if_gather = max(0.0, state.energy - 0.06)
    energy_if_move = max(0.0, state.energy - 0.05)

    prompt = f"""You are an agent in a survival world with calculator tool access.

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
- Energy decays -0.02/step automatically
- Death at energy<=0 or health<=0
- Food consumed 1/10 steps (next consumption in {steps_until_food_consume} steps)
- rest: +0.03 net energy
- gather_resources: -0.06 net energy, +1-3 food
- move_to X Y: -0.05 net energy
- Cooperative gather: both within 2 units = +5 food each

CALCULATOR TOOL
===============
You may use <CALC>expression</CALC> to compute values.
Examples:
  <CALC>0.64 - 0.06</CALC> -> 0.58
  <CALC>5 - 1</CALC> -> 4

PRE-COMPUTED VALUES:
- Energy if rest: {energy_if_rest:.2f}
- Energy if gather: {energy_if_gather:.2f}
- Energy if move: {energy_if_move:.2f}
- Steps until food consumption: {steps_until_food_consume}

RESPONSE FORMAT
===============
<CALC>optional calculation</CALC>
<ACTION>
rest | gather_resources | move_to X Y
</ACTION>

Respond now:"""

    return prompt


# =============================================================================
# MEMORY CONDITION - History Block Builder
# =============================================================================

def build_history_block(history: List[V6HistoryEntry], max_entries: int = 5) -> str:
    """
    Build history block for memory condition.

    V6 adds energy_after and food_after for better tracking.
    """
    if not history:
        return "HISTORY: (no prior steps)"

    # Take last N entries
    recent = history[-max_entries:]

    lines = ["RECENT HISTORY (last {} steps)".format(len(recent))]
    lines.append("=" * 40)

    for entry in recent:
        lines.append(f"Step {entry.step}:")
        lines.append(f"  Before: energy={entry.observation.energy:.2f}, food={entry.observation.food}")
        lines.append(f"  Action: {entry.action}")
        lines.append(f"  After: energy={entry.energy_after:.2f}, food={entry.food_after}")
        lines.append(f"  Reward: {entry.reward:.3f}")
        lines.append(f"  Distance to other: {entry.distance_to_other:.1f}")
        lines.append("")

    return "\n".join(lines)


def build_prompt_with_memory(
    state: V6State,
    history: List[V6HistoryEntry],
    prompt_type: str = "baseline",
    max_history: int = 5
) -> str:
    """
    Build prompt with memory (history) prepended.

    Args:
        state: Current state
        history: List of past observations/actions/rewards
        prompt_type: One of "baseline", "explicit", "reasoning", "tool"
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
    elif prompt_type == "tool":
        base_prompt = build_tool_prompt(state)
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


def parse_tool_response(response: str) -> tuple[Optional[Dict[str, Any]], str, str, List[str]]:
    """
    Parse response from tool prompt.

    Extracts CALC and ACTION sections.

    Returns:
        (action_dict, calc_text, action_text, calc_results)
    """
    import re

    # Extract CALC sections (may be multiple)
    calc_matches = re.findall(r"<CALC>(.*?)</CALC>", response, re.DOTALL | re.IGNORECASE)
    calc_text = "; ".join(calc_matches) if calc_matches else ""

    # Evaluate calculations (safely)
    calc_results = []
    for calc in calc_matches:
        try:
            # Only allow basic arithmetic
            result = eval(calc.strip(), {"__builtins__": {}}, {})
            calc_results.append(f"{calc.strip()} = {result}")
        except:
            calc_results.append(f"{calc.strip()} = ERROR")

    # Extract ACTION section
    action_match = re.search(r"<ACTION>(.*?)</ACTION>", response, re.DOTALL | re.IGNORECASE)
    action_text = action_match.group(1).strip() if action_match else response.strip()

    # Parse the action
    action_dict = parse_baseline_response(action_text)

    return action_dict, calc_text, action_text, calc_results


# =============================================================================
# V6 SCAFFOLD CONDITIONS
# =============================================================================

V6_SCAFFOLDS = [
    "baseline_nomem",      # A: Simple prompt, no memory
    "baseline_memory",     # B: Simple prompt + 5-step history
    "explicit_nomem",      # C: Explicit format, no memory
    "explicit_memory",     # D: Explicit format + history
    "reasoning_nomem",     # E: CoT format, no memory
    "reasoning_memory",    # F: CoT format + history
    "tool_nomem",          # G: Tool support (calculator)
]


def get_scaffold_config(scaffold: str) -> Dict[str, Any]:
    """
    Get configuration for a scaffold condition.

    Returns:
        Dict with prompt_format, memory_enabled, tool_enabled
    """
    configs = {
        "baseline_nomem": {"prompt_format": "baseline", "memory_enabled": False, "tool_enabled": False},
        "baseline_memory": {"prompt_format": "baseline", "memory_enabled": True, "tool_enabled": False},
        "explicit_nomem": {"prompt_format": "explicit", "memory_enabled": False, "tool_enabled": False},
        "explicit_memory": {"prompt_format": "explicit", "memory_enabled": True, "tool_enabled": False},
        "reasoning_nomem": {"prompt_format": "reasoning", "memory_enabled": False, "tool_enabled": False},
        "reasoning_memory": {"prompt_format": "reasoning", "memory_enabled": True, "tool_enabled": False},
        "tool_nomem": {"prompt_format": "tool", "memory_enabled": False, "tool_enabled": True},
    }

    if scaffold not in configs:
        raise ValueError(f"Unknown scaffold: {scaffold}. Valid: {list(configs.keys())}")

    return configs[scaffold]
