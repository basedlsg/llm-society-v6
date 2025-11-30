"""
Experiment type definitions for LLM Society research.

Each experiment type defines:
- Hypothesis to test
- Independent/dependent variables
- Metrics to collect
- Analysis methods
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime
import json


class ExperimentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ANALYZING = "analyzing"


@dataclass
class ExperimentResult:
    """Results from a completed experiment."""
    experiment_id: str
    experiment_type: str
    hypothesis: str
    start_time: datetime
    end_time: datetime
    config_used: Dict[str, Any]

    # Raw data
    raw_metrics: Dict[str, List[float]]
    event_counts: Dict[str, int]
    agent_trajectories: List[Dict[str, Any]]

    # Analyzed results
    summary_statistics: Dict[str, Dict[str, float]]
    correlations: Dict[str, float]
    findings: List[str]
    conclusion: str

    # Meta
    num_agents: int
    num_steps: int
    run_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_type": self.experiment_type,
            "hypothesis": self.hypothesis,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "config_used": self.config_used,
            "raw_metrics": self.raw_metrics,
            "event_counts": self.event_counts,
            "agent_trajectories": self.agent_trajectories,
            "summary_statistics": self.summary_statistics,
            "correlations": self.correlations,
            "findings": self.findings,
            "conclusion": self.conclusion,
            "num_agents": self.num_agents,
            "num_steps": self.num_steps,
            "run_ids": self.run_ids,
        }

    def to_json(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class Experiment:
    """Base experiment definition."""
    name: str
    hypothesis: str
    description: str

    # Experimental design
    independent_variables: Dict[str, List[Any]]
    dependent_variables: List[str]
    control_conditions: Dict[str, Any]

    # Configuration
    base_config_overrides: Dict[str, Any] = field(default_factory=dict)
    num_replications: int = 3
    steps_per_run: int = 50
    agents_per_run: int = 10

    # Metrics to collect
    metrics: List[str] = field(default_factory=list)

    def get_run_configs(self) -> List[Dict[str, Any]]:
        """Generate all configuration combinations for the experiment."""
        configs = []

        # Generate factorial design
        from itertools import product

        var_names = list(self.independent_variables.keys())
        var_values = list(self.independent_variables.values())

        for combination in product(*var_values):
            for rep in range(self.num_replications):
                config = {
                    **self.base_config_overrides,
                    "experiment_name": self.name,
                    "replication": rep + 1,
                }
                for name, value in zip(var_names, combination):
                    config[name] = value
                configs.append(config)

        return configs


@dataclass
class SocialDynamicsExperiment(Experiment):
    """
    Experiment to study how social networks form and evolve.

    Research Questions:
    - How do LLM agents form social connections?
    - What factors influence friendship formation?
    - Do social clusters emerge naturally?
    """

    def __init__(self):
        super().__init__(
            name="social_dynamics",
            hypothesis="LLM agents will form clustered social networks based on "
                      "shared traits and proximity, with hub agents emerging naturally",
            description="Studies the emergence of social structures in LLM agent populations",
            independent_variables={
                "agent_count": [5, 10, 20],
                "social_radius": [5.0, 10.0, 20.0],
                "world_size": [(50, 50), (100, 100)],
            },
            dependent_variables=[
                "network_density",
                "clustering_coefficient",
                "avg_connection_strength",
                "num_isolated_agents",
                "max_degree_centrality",
            ],
            control_conditions={
                "movement_speed": 1.0,
                "interaction_radius": 5.0,
            },
            num_replications=3,
            steps_per_run=30,
            agents_per_run=10,
            metrics=[
                "social_connections_formed",
                "social_interactions_count",
                "avg_happiness",
                "network_modularity",
            ],
        )


@dataclass
class EconomicBehaviorExperiment(Experiment):
    """
    Experiment to study economic decision-making and market dynamics.

    Research Questions:
    - How do agents allocate resources?
    - Do markets reach equilibrium?
    - What trading strategies emerge?
    """

    def __init__(self):
        super().__init__(
            name="economic_behavior",
            hypothesis="LLM agents will develop rational trading strategies and "
                      "markets will trend toward price equilibrium over time",
            description="Studies economic behavior and market dynamics in LLM societies",
            independent_variables={
                "agent_count": [6, 12],
                "initial_resource_variance": ["low", "high"],
                "market_visibility": [True, False],
            },
            dependent_variables=[
                "price_volatility",
                "wealth_gini_coefficient",
                "market_volume",
                "resource_distribution_entropy",
            ],
            control_conditions={
                "initial_currency": 100.0,
            },
            num_replications=3,
            steps_per_run=40,
            agents_per_run=8,
            metrics=[
                "transactions_count",
                "avg_transaction_size",
                "wealth_distribution",
                "resource_prices",
            ],
        )


@dataclass
class EmergentCultureExperiment(Experiment):
    """
    Experiment to study cultural dynamics and group formation.

    Research Questions:
    - Do distinct cultural groups emerge?
    - How do cultural affinities spread?
    - What drives cultural convergence/divergence?
    """

    def __init__(self):
        super().__init__(
            name="emergent_culture",
            hypothesis="LLM agents will self-organize into cultural groups with "
                      "distinct behavioral patterns and shared values",
            description="Studies the emergence of cultural groups and value systems",
            independent_variables={
                "agent_count": [10, 20],
                "initial_cultural_diversity": ["homogeneous", "diverse"],
                "interaction_frequency": ["low", "high"],
            },
            dependent_variables=[
                "num_cultural_clusters",
                "cultural_homogeneity_index",
                "inter_group_interaction_rate",
                "value_convergence_rate",
            ],
            control_conditions={
                "world_size": (100, 100),
            },
            num_replications=3,
            steps_per_run=50,
            agents_per_run=15,
            metrics=[
                "cultural_affinity_changes",
                "group_membership_stability",
                "cross_cultural_interactions",
            ],
        )


@dataclass
class CooperationExperiment(Experiment):
    """
    Experiment to study cooperation and collective behavior.

    Research Questions:
    - When do agents cooperate vs compete?
    - How does reputation affect cooperation?
    - Can collective goals emerge?
    """

    def __init__(self):
        super().__init__(
            name="cooperation_dynamics",
            hypothesis="LLM agents will develop reputation-based cooperation strategies, "
                      "with higher cooperation rates among agents with shared history",
            description="Studies cooperation patterns and collective behavior emergence",
            independent_variables={
                "agent_count": [6, 12],
                "resource_scarcity": ["abundant", "scarce"],
                "reputation_visibility": [True, False],
            },
            dependent_variables=[
                "cooperation_rate",
                "defection_rate",
                "reputation_correlation",
                "collective_welfare",
            ],
            control_conditions={
                "memory_size": 20,
            },
            num_replications=3,
            steps_per_run=40,
            agents_per_run=8,
            metrics=[
                "help_actions_count",
                "resource_sharing_events",
                "reputation_changes",
                "group_task_completions",
            ],
        )
