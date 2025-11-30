"""
Heuristic Policy for v1.1 Baseline Comparison

This implements a simple rule-based survival policy that serves as
a baseline for comparing LLM behavior.

Rules (in priority order):
1. If energy < 0.3: rest
2. If food <= 2: gather_resources
3. If nearby agent within social_radius: talk_to (30% chance)
4. Otherwise: move towards center or random exploration
"""

import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class HeuristicConfig:
    """Configuration for the heuristic policy."""
    energy_rest_threshold: float = 0.3  # Rest if energy below this
    food_gather_threshold: int = 2  # Gather if food at or below this
    talk_probability: float = 0.3  # Probability to talk when agent nearby
    explore_probability: float = 0.4  # Probability to move randomly


def heuristic_decide(
    energy: float,
    health: float,
    food: int,
    position: tuple,
    nearby_agents: List[str],
    social_connections: Dict[str, float],
    world_size: tuple,
    config: Optional[HeuristicConfig] = None,
) -> Dict[str, Any]:
    """
    Make a survival-oriented decision based on simple rules.

    Args:
        energy: Current energy level (0-1)
        health: Current health level (0-1)
        food: Current food count
        position: (x, y) position
        nearby_agents: List of agent IDs within social_radius
        social_connections: Dict of existing social connections
        world_size: (width, height) of the world
        config: Optional configuration overrides

    Returns:
        Dict with 'type' and 'params' for the action
    """
    if config is None:
        config = HeuristicConfig()

    # Rule 1: Rest if energy is critically low
    if energy < config.energy_rest_threshold:
        return {"type": "rest", "params": {}, "heuristic_rule": "low_energy"}

    # Rule 2: Gather if food is low (survival priority)
    if food <= config.food_gather_threshold:
        return {"type": "gather_resources", "params": {}, "heuristic_rule": "low_food"}

    # Rule 3: Talk to nearby agents with some probability
    if nearby_agents and random.random() < config.talk_probability:
        # Prefer agents we haven't connected with yet
        unconnected = [a for a in nearby_agents if a not in social_connections]
        if unconnected:
            target = random.choice(unconnected)
        else:
            target = random.choice(nearby_agents)
        return {"type": "talk_to", "params": {"target_id": target}, "heuristic_rule": "social"}

    # Rule 4: Explore / move randomly
    if random.random() < config.explore_probability:
        # Move towards a random position in the world
        new_x = random.uniform(0, world_size[0])
        new_y = random.uniform(0, world_size[1])
        return {"type": "move_to", "params": {"x": new_x, "y": new_y}, "heuristic_rule": "explore"}

    # Default: rest to conserve energy
    return {"type": "rest", "params": {}, "heuristic_rule": "default"}


class HeuristicAgent:
    """
    A wrapper that applies heuristic policy to an LLMAgent.

    This can be used to replace LLM decision-making with rule-based decisions
    for baseline comparison experiments.
    """

    def __init__(self, config: Optional[HeuristicConfig] = None):
        self.config = config or HeuristicConfig()
        self.decisions_made = 0
        self.decision_log = []

    def decide(
        self,
        agent_state: Dict[str, Any],
        nearby_agents: List[str],
        world_size: tuple,
    ) -> Dict[str, Any]:
        """
        Make a decision for the given agent state.

        Args:
            agent_state: Dict with energy, health, food, position, social_connections
            nearby_agents: List of nearby agent IDs
            world_size: (width, height) of the world

        Returns:
            Action dict with type, params, and source="heuristic"
        """
        action = heuristic_decide(
            energy=agent_state.get("energy", 1.0),
            health=agent_state.get("health", 1.0),
            food=agent_state.get("food", 5),
            position=agent_state.get("position", (10, 10)),
            nearby_agents=nearby_agents,
            social_connections=agent_state.get("social_connections", {}),
            world_size=world_size,
            config=self.config,
        )

        action["source"] = "heuristic"
        self.decisions_made += 1
        self.decision_log.append(action)

        return action

    def get_action_distribution(self) -> Dict[str, int]:
        """Get counts of each action type chosen."""
        distribution = {}
        for decision in self.decision_log:
            action_type = decision.get("type", "unknown")
            distribution[action_type] = distribution.get(action_type, 0) + 1
        return distribution


def run_heuristic_simulation_step(agent, heuristic_agent: HeuristicAgent):
    """
    Run a single step using heuristic policy instead of LLM.

    This function can be called from the agent's step() method when
    running in heuristic mode.
    """
    # Get agent state
    agent_state = {
        "energy": agent.energy,
        "health": agent.health,
        "food": agent.resources.get("food", 0),
        "position": (agent.position.x, agent.position.y),
        "social_connections": agent.social_connections,
    }

    # Get nearby agents
    nearby_agents = [a.unique_id for a in agent._get_nearby_agents()]

    # Get world size
    world_size = (
        agent.config.simulation.world_size[0],
        agent.config.simulation.world_size[1],
    )

    # Make decision
    return heuristic_agent.decide(agent_state, nearby_agents, world_size)
