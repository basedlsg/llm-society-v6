"""
Behavioral Fingerprint for LLM Society v1.0

A behavioral fingerprint captures the characteristic patterns
of how a model (or population) behaves in the simulation.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Counter
from collections import Counter as PyCounter
import statistics
import json

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


@dataclass
class BehavioralFingerprint:
    """
    A vector of metrics characterizing model behavior.

    This fingerprint allows comparison between:
    - Different models under same conditions
    - Same model under different conditions
    - Different runs for stability analysis
    """

    # Network metrics
    mean_degree: float
    isolation_fraction: float
    clustering_coefficient: float
    max_degree: int

    # Economic metrics
    gini_currency: float
    mean_food: float
    starvation_rate: float
    trade_frequency: float

    # Survival metrics
    survival_rate: float
    mean_time_to_death: Optional[float]

    # Activity metrics
    rest_fraction: float
    move_fraction: float
    talk_fraction: float
    gather_fraction: float
    trade_fraction: float

    # Happiness metrics (analysis only)
    mean_happiness: float
    happiness_std: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BehavioralFingerprint":
        return cls(**data)

    def compare_to(self, other: "BehavioralFingerprint") -> Dict[str, Dict[str, float]]:
        """
        Compare this fingerprint to another.

        Returns dict with differences for each metric.
        """
        differences = {}

        for field_name in self.__dataclass_fields__:
            val_self = getattr(self, field_name)
            val_other = getattr(other, field_name)

            if val_self is not None and val_other is not None:
                diff = val_other - val_self
                pct_change = (diff / val_self * 100) if val_self != 0 else float('inf')
                differences[field_name] = {
                    "self": val_self,
                    "other": val_other,
                    "diff": diff,
                    "pct_change": pct_change,
                }

        return differences


def compute_gini(values: List[float]) -> float:
    """
    Compute Gini coefficient for inequality measurement.

    Args:
        values: List of values (e.g., currency amounts)

    Returns:
        Gini coefficient in [0, 1] where 0 = perfect equality
    """
    if not values or len(values) < 2:
        return 0.0

    if sum(values) == 0:
        return 0.0

    sorted_values = sorted(values)
    n = len(sorted_values)
    cumsum = sum((i + 1) * v for i, v in enumerate(sorted_values))

    return (2 * cumsum) / (n * sum(sorted_values)) - (n + 1) / n


def compute_degree_distribution(agents) -> Dict[str, Any]:
    """
    Compute degree distribution from agents.

    Args:
        agents: List of agent objects or dicts

    Returns:
        Dict with degree statistics
    """
    degrees = []

    for agent in agents:
        if isinstance(agent, dict):
            connections = agent.get("social_connections", {})
            if isinstance(connections, dict):
                degree = len(connections)
            elif isinstance(connections, (int, float)):
                degree = int(connections)
            else:
                degree = agent.get("num_connections", 0)
        else:
            if hasattr(agent, 'social_connections'):
                degree = len(agent.social_connections)
            else:
                degree = 0
        degrees.append(degree)

    if not degrees:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
            "distribution": {},
        }

    return {
        "mean": statistics.mean(degrees),
        "std": statistics.stdev(degrees) if len(degrees) > 1 else 0.0,
        "min": min(degrees),
        "max": max(degrees),
        "distribution": dict(PyCounter(degrees)),
    }


def compute_clustering_coefficient(agents) -> float:
    """
    Compute average clustering coefficient.

    Requires networkx. Returns 0.0 if not available.
    """
    if not NETWORKX_AVAILABLE:
        return 0.0

    # Build graph
    G = nx.Graph()

    for agent in agents:
        if isinstance(agent, dict):
            agent_id = agent.get("agent_id", agent.get("unique_id", str(id(agent))))
            connections = agent.get("social_connections", {})
        else:
            agent_id = getattr(agent, 'unique_id', str(id(agent)))
            connections = getattr(agent, 'social_connections', {})

        G.add_node(agent_id)

        if isinstance(connections, dict):
            for connected_id in connections:
                G.add_edge(agent_id, connected_id)

    if G.number_of_nodes() == 0:
        return 0.0

    return nx.average_clustering(G)


def compute_action_distribution(events: List[Dict]) -> Dict[str, float]:
    """
    Compute distribution of action types.

    Args:
        events: List of event dictionaries

    Returns:
        Dict mapping action types to fractions
    """
    action_counts = PyCounter()
    total = 0

    for event in events:
        if event.get("event_type") == "AGENT_ACTION_CHOSEN":
            details = event.get("details", {})
            action_type = details.get("action_type", "unknown")
            action_counts[action_type] += 1
            total += 1

    if total == 0:
        return {
            "rest": 0.0,
            "move_to": 0.0,
            "talk_to": 0.0,
            "gather_resources": 0.0,
            "market_trade": 0.0,
        }

    return {
        action: count / total
        for action, count in action_counts.items()
    }


def compute_behavioral_fingerprint(
    agents,
    events: List[Dict],
    death_events: Optional[List[Dict]] = None,
    happiness_values: Optional[List[float]] = None,
) -> BehavioralFingerprint:
    """
    Compute a complete behavioral fingerprint.

    Args:
        agents: List of agent objects or dicts
        events: List of simulation events
        death_events: Optional list of death events
        happiness_values: Optional pre-computed happiness values

    Returns:
        BehavioralFingerprint dataclass
    """
    # Network metrics
    degree_dist = compute_degree_distribution(agents)
    mean_degree = degree_dist["mean"]
    max_degree = degree_dist["max"]

    # Isolation fraction
    degrees = []
    for agent in agents:
        if isinstance(agent, dict):
            connections = agent.get("social_connections", {})
            if isinstance(connections, dict):
                degrees.append(len(connections))
            else:
                degrees.append(agent.get("num_connections", 0))
        else:
            degrees.append(len(getattr(agent, 'social_connections', {})))

    isolation_fraction = sum(1 for d in degrees if d == 0) / len(degrees) if degrees else 0.0

    # Clustering
    clustering_coefficient = compute_clustering_coefficient(agents)

    # Economic metrics
    currency_values = []
    food_values = []
    for agent in agents:
        if isinstance(agent, dict):
            resources = agent.get("resources", {})
            currency_values.append(resources.get("currency", agent.get("currency", 0)))
            food_values.append(resources.get("food", agent.get("food", 0)))
        else:
            resources = getattr(agent, 'resources', {})
            currency_values.append(resources.get("currency", 0))
            food_values.append(resources.get("food", 0))

    gini_currency = compute_gini(currency_values)
    mean_food = statistics.mean(food_values) if food_values else 0.0
    starvation_rate = sum(1 for f in food_values if f == 0) / len(food_values) if food_values else 0.0

    # Activity metrics
    action_dist = compute_action_distribution(events)
    rest_fraction = action_dist.get("rest", 0.0)
    move_fraction = action_dist.get("move_to", 0.0)
    talk_fraction = action_dist.get("talk_to", 0.0)
    gather_fraction = action_dist.get("gather_resources", 0.0)
    trade_fraction = action_dist.get("market_trade", 0.0)

    # Survival metrics
    alive_count = 0
    for agent in agents:
        if isinstance(agent, dict):
            alive = agent.get("alive", True)
        else:
            alive = getattr(agent, 'alive', True)
        if alive:
            alive_count += 1

    survival_rate = alive_count / len(agents) if agents else 0.0

    # Time to death
    mean_time_to_death = None
    if death_events:
        death_times = [e.get("step", 0) for e in death_events]
        if death_times:
            mean_time_to_death = statistics.mean(death_times)

    # Happiness metrics
    if happiness_values:
        mean_happiness = statistics.mean(happiness_values)
        happiness_std = statistics.stdev(happiness_values) if len(happiness_values) > 1 else 0.0
    else:
        # Compute happiness if not provided
        from .happiness import HappinessCalculator, calculate_population_happiness
        pop_happiness = calculate_population_happiness(agents)
        mean_happiness = pop_happiness["mean"]
        happiness_std = pop_happiness["std"]

    return BehavioralFingerprint(
        mean_degree=mean_degree,
        isolation_fraction=isolation_fraction,
        clustering_coefficient=clustering_coefficient,
        max_degree=max_degree,
        gini_currency=gini_currency,
        mean_food=mean_food,
        starvation_rate=starvation_rate,
        trade_frequency=trade_fraction,
        survival_rate=survival_rate,
        mean_time_to_death=mean_time_to_death,
        rest_fraction=rest_fraction,
        move_fraction=move_fraction,
        talk_fraction=talk_fraction,
        gather_fraction=gather_fraction,
        trade_fraction=trade_fraction,
        mean_happiness=mean_happiness,
        happiness_std=happiness_std,
    )
