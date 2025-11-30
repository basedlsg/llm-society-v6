"""
Invisible Happiness Metric for LLM Society v1.0

IMPORTANT: This metric is NEVER exposed to agents.
It is computed purely for analysis and research purposes.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

# Default happiness weights
HAPPINESS_WEIGHTS = {
    "w_social": 0.4,
    "w_resource": 0.3,
    "w_health": 0.3,
}

# Saturation points for normalization
HAPPINESS_PARAMS = {
    "K_social": 10,      # Saturates after 10 connections
    "F_safe": 20,        # "Safe" food level
    "C_safe": 1000,      # "Safe" currency level
}


@dataclass
class HappinessComponents:
    """Breakdown of happiness into component terms."""
    social_term: float
    resource_term: float
    health_term: float
    happiness: float


class HappinessCalculator:
    """
    Calculates the invisible happiness metric for agents.

    This metric is NEVER shown to agents. It exists purely for:
    - Post-simulation analysis
    - Model comparison research
    - Behavioral fingerprinting

    The happiness formula:
        happiness = w_social * social_term
                  + w_resource * resource_term
                  + w_health * health_term

    Where:
        social_term = min(1.0, degree / K_social)
        resource_term = 0.5 * min(1.0, food/F_safe) + 0.5 * min(1.0, currency/C_safe)
        health_term = health (already in [0,1])
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, float]] = None,
    ):
        self.weights = weights or HAPPINESS_WEIGHTS.copy()
        self.params = params or HAPPINESS_PARAMS.copy()

    def social_term(self, agent) -> float:
        """
        Calculate social term based on number of connections.

        Args:
            agent: Agent object with social_connections attribute

        Returns:
            float in [0, 1]
        """
        if hasattr(agent, 'social_connections'):
            degree = len(agent.social_connections)
        else:
            degree = 0

        return min(1.0, degree / self.params["K_social"])

    def resource_term(self, agent) -> float:
        """
        Calculate resource safety term.

        Args:
            agent: Agent object with food and currency attributes

        Returns:
            float in [0, 1]
        """
        # Get food
        if hasattr(agent, 'resources') and isinstance(agent.resources, dict):
            food = agent.resources.get("food", 0)
            currency = agent.resources.get("currency", 0)
        elif hasattr(agent, 'food') and hasattr(agent, 'currency'):
            food = agent.food
            currency = agent.currency
        else:
            food = 0
            currency = 0

        food_norm = min(1.0, food / self.params["F_safe"])
        cash_norm = min(1.0, currency / self.params["C_safe"])

        return 0.5 * food_norm + 0.5 * cash_norm

    def health_term(self, agent) -> float:
        """
        Calculate health term.

        Args:
            agent: Agent object with health attribute

        Returns:
            float in [0, 1]
        """
        if hasattr(agent, 'health'):
            return max(0.0, min(1.0, agent.health))
        return 1.0  # Default to full health if not available

    def calculate(self, agent) -> float:
        """
        Calculate the happiness metric for an agent.

        Args:
            agent: Agent object

        Returns:
            float in [0, 1] representing happiness
        """
        social = self.social_term(agent)
        resource = self.resource_term(agent)
        health = self.health_term(agent)

        happiness = (
            self.weights["w_social"] * social +
            self.weights["w_resource"] * resource +
            self.weights["w_health"] * health
        )

        return happiness

    def calculate_with_components(self, agent) -> HappinessComponents:
        """
        Calculate happiness with full component breakdown.

        Args:
            agent: Agent object

        Returns:
            HappinessComponents dataclass with all terms
        """
        social = self.social_term(agent)
        resource = self.resource_term(agent)
        health = self.health_term(agent)

        happiness = (
            self.weights["w_social"] * social +
            self.weights["w_resource"] * resource +
            self.weights["w_health"] * health
        )

        return HappinessComponents(
            social_term=social,
            resource_term=resource,
            health_term=health,
            happiness=happiness,
        )

    def calculate_from_dict(self, agent_state: Dict[str, Any]) -> float:
        """
        Calculate happiness from a dictionary representation.

        Args:
            agent_state: Dictionary with agent state

        Returns:
            float in [0, 1]
        """
        # Social term
        connections = agent_state.get("social_connections", {})
        if isinstance(connections, dict):
            degree = len(connections)
        elif isinstance(connections, (int, float)):
            degree = int(connections)
        else:
            degree = 0
        social = min(1.0, degree / self.params["K_social"])

        # Resource term
        resources = agent_state.get("resources", {})
        if isinstance(resources, dict):
            food = resources.get("food", agent_state.get("food", 0))
            currency = resources.get("currency", agent_state.get("currency", 0))
        else:
            food = agent_state.get("food", 0)
            currency = agent_state.get("currency", 0)

        food_norm = min(1.0, food / self.params["F_safe"])
        cash_norm = min(1.0, currency / self.params["C_safe"])
        resource = 0.5 * food_norm + 0.5 * cash_norm

        # Health term
        health = agent_state.get("health", 1.0)
        health = max(0.0, min(1.0, health))

        # Combined
        happiness = (
            self.weights["w_social"] * social +
            self.weights["w_resource"] * resource +
            self.weights["w_health"] * health
        )

        return happiness


def calculate_population_happiness(agents, calculator: Optional[HappinessCalculator] = None):
    """
    Calculate happiness statistics for a population.

    Args:
        agents: List of agent objects or dicts
        calculator: HappinessCalculator instance (uses default if None)

    Returns:
        Dict with mean, std, min, max, and individual values
    """
    if calculator is None:
        calculator = HappinessCalculator()

    happiness_values = []

    for agent in agents:
        if isinstance(agent, dict):
            h = calculator.calculate_from_dict(agent)
        else:
            h = calculator.calculate(agent)
        happiness_values.append(h)

    if not happiness_values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "values": [],
        }

    import statistics

    return {
        "mean": statistics.mean(happiness_values),
        "std": statistics.stdev(happiness_values) if len(happiness_values) > 1 else 0.0,
        "min": min(happiness_values),
        "max": max(happiness_values),
        "values": happiness_values,
    }
