"""
Metrics Collector for LLM Society v1.0

Collects and stores metrics throughout simulation runs.
"""

import json
import logging
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .happiness import HappinessCalculator, HAPPINESS_WEIGHTS
from .behavioral import BehavioralFingerprint, compute_behavioral_fingerprint

logger = logging.getLogger(__name__)


@dataclass
class AgentMetricsSnapshot:
    """Snapshot of agent metrics at a single step."""
    agent_id: str
    step: int
    happiness: float
    social_term: float
    resource_term: float
    health_term: float
    degree: int
    food: int
    currency: int
    materials: int
    health: float
    energy: float
    alive: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PopulationMetrics:
    """Population-level metrics at a single step."""
    step: int
    happiness_mean: float
    happiness_std: float
    happiness_min: float
    happiness_max: float
    alive_count: int
    total_agents: int
    mean_degree: float
    mean_energy: float
    mean_food: float
    mean_currency: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricsCollector:
    """
    Collects metrics throughout a simulation run.

    Features:
    - Per-agent happiness calculation (invisible to agents)
    - Population statistics per step
    - Event tracking
    - Behavioral fingerprint generation
    """

    def __init__(
        self,
        happiness_weights: Optional[Dict[str, float]] = None,
        output_dir: Optional[str] = None,
    ):
        self.happiness_calculator = HappinessCalculator(weights=happiness_weights)
        self.output_dir = Path(output_dir) if output_dir else None

        # Storage
        self.step_metrics: List[Dict[str, Any]] = []
        self.agent_trajectories: Dict[str, List[AgentMetricsSnapshot]] = {}
        self.population_trajectory: List[PopulationMetrics] = []
        self.events: List[Dict[str, Any]] = []
        self.death_events: List[Dict[str, Any]] = []

        # Metadata
        self.run_id: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.config_snapshot: Optional[Dict] = None

    def start_run(self, run_id: str, config: Optional[Dict] = None):
        """Initialize a new collection run."""
        self.run_id = run_id
        self.start_time = datetime.now()
        self.config_snapshot = config
        self.step_metrics = []
        self.agent_trajectories = {}
        self.population_trajectory = []
        self.events = []
        self.death_events = []
        logger.info(f"MetricsCollector started run: {run_id}")

    def end_run(self):
        """Finalize the collection run."""
        self.end_time = datetime.now()
        logger.info(f"MetricsCollector ended run: {self.run_id}")

    def collect_step(self, step: int, agents, events: Optional[List[Dict]] = None):
        """
        Collect all metrics for a single simulation step.

        Args:
            step: Current simulation step
            agents: List of agent objects
            events: Optional list of events from this step
        """
        agent_snapshots = []

        for agent in agents:
            # Get agent ID
            if isinstance(agent, dict):
                agent_id = agent.get("agent_id", agent.get("unique_id", str(id(agent))))
            else:
                agent_id = getattr(agent, 'unique_id', str(id(agent)))

            # Calculate happiness components
            components = self.happiness_calculator.calculate_with_components(agent)

            # Get agent state
            if isinstance(agent, dict):
                resources = agent.get("resources", {})
                connections = agent.get("social_connections", {})
                snapshot = AgentMetricsSnapshot(
                    agent_id=agent_id,
                    step=step,
                    happiness=components.happiness,
                    social_term=components.social_term,
                    resource_term=components.resource_term,
                    health_term=components.health_term,
                    degree=len(connections) if isinstance(connections, dict) else int(connections),
                    food=resources.get("food", agent.get("food", 0)),
                    currency=resources.get("currency", agent.get("currency", 0)),
                    materials=resources.get("materials", agent.get("materials", 0)),
                    health=agent.get("health", 1.0),
                    energy=agent.get("energy", 1.0),
                    alive=agent.get("alive", True),
                )
            else:
                resources = getattr(agent, 'resources', {})
                connections = getattr(agent, 'social_connections', {})
                snapshot = AgentMetricsSnapshot(
                    agent_id=agent_id,
                    step=step,
                    happiness=components.happiness,
                    social_term=components.social_term,
                    resource_term=components.resource_term,
                    health_term=components.health_term,
                    degree=len(connections),
                    food=resources.get("food", 0),
                    currency=resources.get("currency", 0),
                    materials=resources.get("materials", 0),
                    health=getattr(agent, 'health', 1.0),
                    energy=getattr(agent, 'energy', 1.0),
                    alive=getattr(agent, 'alive', True),
                )

            agent_snapshots.append(snapshot)

            # Track trajectory
            if agent_id not in self.agent_trajectories:
                self.agent_trajectories[agent_id] = []
            self.agent_trajectories[agent_id].append(snapshot)

        # Population statistics
        happiness_values = [s.happiness for s in agent_snapshots]
        degrees = [s.degree for s in agent_snapshots]
        energies = [s.energy for s in agent_snapshots]
        foods = [s.food for s in agent_snapshots]
        currencies = [s.currency for s in agent_snapshots]

        pop_metrics = PopulationMetrics(
            step=step,
            happiness_mean=statistics.mean(happiness_values) if happiness_values else 0.0,
            happiness_std=statistics.stdev(happiness_values) if len(happiness_values) > 1 else 0.0,
            happiness_min=min(happiness_values) if happiness_values else 0.0,
            happiness_max=max(happiness_values) if happiness_values else 0.0,
            alive_count=sum(1 for s in agent_snapshots if s.alive),
            total_agents=len(agent_snapshots),
            mean_degree=statistics.mean(degrees) if degrees else 0.0,
            mean_energy=statistics.mean(energies) if energies else 0.0,
            mean_food=statistics.mean(foods) if foods else 0.0,
            mean_currency=statistics.mean(currencies) if currencies else 0.0,
        )

        self.population_trajectory.append(pop_metrics)

        # Store step data
        self.step_metrics.append({
            "step": step,
            "agents": [s.to_dict() for s in agent_snapshots],
            "population": pop_metrics.to_dict(),
        })

        # Track events
        if events:
            for event in events:
                event["collected_step"] = step
                self.events.append(event)

                # Track deaths
                if event.get("event_type") == "AGENT_DEATH":
                    self.death_events.append(event)

    def record_event(self, event: Dict[str, Any]):
        """Record a single event."""
        self.events.append(event)
        if event.get("event_type") == "AGENT_DEATH":
            self.death_events.append(event)

    def get_happiness_trajectory(self) -> List[Dict[str, float]]:
        """Get happiness over time (population level)."""
        return [
            {
                "step": pm.step,
                "mean": pm.happiness_mean,
                "std": pm.happiness_std,
                "min": pm.happiness_min,
                "max": pm.happiness_max,
            }
            for pm in self.population_trajectory
        ]

    def get_agent_happiness_trajectory(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get happiness trajectory for a specific agent."""
        if agent_id not in self.agent_trajectories:
            return []

        return [
            {
                "step": s.step,
                "happiness": s.happiness,
                "social_term": s.social_term,
                "resource_term": s.resource_term,
                "health_term": s.health_term,
            }
            for s in self.agent_trajectories[agent_id]
        ]

    def generate_fingerprint(self, agents, events: Optional[List[Dict]] = None) -> BehavioralFingerprint:
        """Generate behavioral fingerprint from collected data."""
        all_events = events or self.events
        happiness_values = [s.happiness_mean for s in self.population_trajectory[-1:]] if self.population_trajectory else None

        # Get final agent states if available
        if self.step_metrics:
            final_agents = self.step_metrics[-1].get("agents", [])
            happiness_values = [a["happiness"] for a in final_agents]
        else:
            final_agents = agents
            happiness_values = None

        return compute_behavioral_fingerprint(
            agents=final_agents if final_agents else agents,
            events=all_events,
            death_events=self.death_events,
            happiness_values=happiness_values,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of collected metrics."""
        if not self.population_trajectory:
            return {"error": "No data collected"}

        final_pop = self.population_trajectory[-1]

        # Happiness over time
        happiness_trajectory = self.get_happiness_trajectory()

        return {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_steps": len(self.population_trajectory),
            "total_agents": final_pop.total_agents,
            "final_alive": final_pop.alive_count,
            "survival_rate": final_pop.alive_count / final_pop.total_agents if final_pop.total_agents > 0 else 0,

            "happiness": {
                "final_mean": final_pop.happiness_mean,
                "final_std": final_pop.happiness_std,
                "final_min": final_pop.happiness_min,
                "final_max": final_pop.happiness_max,
                "trajectory": happiness_trajectory,
            },

            "network": {
                "final_mean_degree": final_pop.mean_degree,
            },

            "economy": {
                "final_mean_food": final_pop.mean_food,
                "final_mean_currency": final_pop.mean_currency,
            },

            "events": {
                "total": len(self.events),
                "deaths": len(self.death_events),
            },
        }

    def export_to_json(self, filepath: str):
        """Export all collected data to JSON."""
        data = {
            "metadata": {
                "run_id": self.run_id,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "config": self.config_snapshot,
            },
            "population_trajectory": [pm.to_dict() for pm in self.population_trajectory],
            "summary": self.get_summary(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported metrics to {filepath}")

    def export_happiness_csv(self, filepath: str):
        """Export happiness trajectory to CSV."""
        import csv

        trajectory = self.get_happiness_trajectory()

        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["step", "mean", "std", "min", "max"])
            writer.writeheader()
            writer.writerows(trajectory)

        logger.info(f"Exported happiness trajectory to {filepath}")
