"""
Experiment Coordinator for LLM Society Research.

This agent orchestrates experiments, collects data, and analyzes results
to extract valuable insights about LLM agent behavior.
"""

import asyncio
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Type
import statistics

from .experiment_types import (
    Experiment,
    ExperimentResult,
    ExperimentStatus,
    SocialDynamicsExperiment,
    EconomicBehaviorExperiment,
    EmergentCultureExperiment,
    CooperationExperiment,
)

logger = logging.getLogger(__name__)


@dataclass
class RunData:
    """Data collected from a single simulation run."""
    run_id: str
    config: Dict[str, Any]
    step_data: List[Dict[str, Any]] = field(default_factory=list)
    final_agent_states: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, List[float]] = field(default_factory=dict)


class ExperimentCoordinator:
    """
    Coordinates experiments on LLM Society simulations.

    Capabilities:
    - Design and run controlled experiments
    - Collect metrics across multiple runs
    - Analyze results and compute statistics
    - Generate research insights
    """

    EXPERIMENT_TYPES: Dict[str, Type[Experiment]] = {
        "social_dynamics": SocialDynamicsExperiment,
        "economic_behavior": EconomicBehaviorExperiment,
        "emergent_culture": EmergentCultureExperiment,
        "cooperation_dynamics": CooperationExperiment,
    }

    def __init__(
        self,
        output_dir: str = "./experiment_results",
        db_path: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path or "./llm_society_dynamic_data.db"
        self.current_experiment: Optional[Experiment] = None
        self.run_data: List[RunData] = []

        logger.info(f"ExperimentCoordinator initialized. Output: {self.output_dir}")

    def list_experiments(self) -> List[str]:
        """List available experiment types."""
        return list(self.EXPERIMENT_TYPES.keys())

    def get_experiment(self, experiment_type: str) -> Experiment:
        """Get an experiment instance by type."""
        if experiment_type not in self.EXPERIMENT_TYPES:
            raise ValueError(
                f"Unknown experiment type: {experiment_type}. "
                f"Available: {list(self.EXPERIMENT_TYPES.keys())}"
            )
        return self.EXPERIMENT_TYPES[experiment_type]()

    async def run_experiment(
        self,
        experiment_type: str,
        num_agents: int = 5,
        num_steps: int = 20,
        num_runs: int = 1,
        model_name: str = "gemini-pro",
    ) -> ExperimentResult:
        """
        Run a complete experiment with multiple simulation runs.

        Args:
            experiment_type: Type of experiment to run
            num_agents: Number of agents per run
            num_steps: Number of simulation steps per run
            num_runs: Number of replications
            model_name: LLM model to use

        Returns:
            ExperimentResult with collected data and analysis
        """
        experiment = self.get_experiment(experiment_type)
        experiment.agents_per_run = num_agents
        experiment.steps_per_run = num_steps
        experiment.num_replications = num_runs

        self.current_experiment = experiment
        self.run_data = []

        experiment_id = f"{experiment_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()

        logger.info(f"Starting experiment: {experiment_id}")
        logger.info(f"Hypothesis: {experiment.hypothesis}")
        logger.info(f"Runs planned: {num_runs} x {num_agents} agents x {num_steps} steps")

        # Import here to avoid circular imports
        from ..simulation.society_simulator import SocietySimulator
        from ..utils.config import Config

        run_ids = []

        for run_idx in range(num_runs):
            logger.info(f"=== Run {run_idx + 1}/{num_runs} ===")

            # Create config for this run
            config = Config.default()
            config.agents.count = num_agents
            config.simulation.max_steps = num_steps
            config.llm.model_name = model_name

            # Apply experiment-specific settings
            for key, value in experiment.control_conditions.items():
                if hasattr(config.agents, key):
                    setattr(config.agents, key, value)
                elif hasattr(config.simulation, key):
                    setattr(config.simulation, key, value)

            # Run simulation
            simulator = SocietySimulator(config)

            try:
                await simulator.run()
                run_id = getattr(simulator.database_handler, '_current_run_id', f"run_{run_idx}")
                run_ids.append(run_id)

                # Collect data from this run
                run_data = self._collect_run_data(simulator, run_id, config)
                self.run_data.append(run_data)

                logger.info(f"Run {run_idx + 1} completed. Run ID: {run_id}")

            except Exception as e:
                logger.error(f"Run {run_idx + 1} failed: {e}")
                continue

            finally:
                if hasattr(simulator, 'llm_coordinator') and simulator.llm_coordinator:
                    await simulator.llm_coordinator.stop()

        end_time = datetime.now()

        # Analyze results
        result = self._analyze_results(
            experiment_id=experiment_id,
            experiment=experiment,
            start_time=start_time,
            end_time=end_time,
            run_ids=run_ids,
            num_agents=num_agents,
            num_steps=num_steps,
        )

        # Save results
        result_path = self.output_dir / f"{experiment_id}_results.json"
        result.to_json(str(result_path))
        logger.info(f"Results saved to: {result_path}")

        return result

    def _collect_run_data(
        self,
        simulator,
        run_id: str,
        config,
    ) -> RunData:
        """Collect data from a completed simulation run."""
        run_data = RunData(
            run_id=run_id,
            config=config.to_dict() if hasattr(config, 'to_dict') else {},
        )

        # Collect final agent states
        for agent in simulator._agent_list:
            agent_state = {
                "agent_id": agent.unique_id,
                "position": {"x": agent.position.x, "y": agent.position.y},
                "energy": agent.energy,
                "happiness": agent.happiness,
                "health": agent.health,
                "age": agent.age,
                "employed": agent.employed,
                "social_connections": dict(agent.social_connections),
                "num_connections": len(agent.social_connections),
                "social_reputation": agent.social_reputation,
                "resources": dict(agent.resources),
                "cultural_group": agent.cultural_group_id,
                "cultural_affinities": dict(agent.cultural_affinities),
                "credit_score": agent.credit_score,
                "total_debt": agent.total_debt,
                "monthly_income": agent.monthly_income,
                "memories_count": len(agent.memories),
                "agent_type": agent.agent_type.name if hasattr(agent.agent_type, 'name') else str(agent.agent_type),
            }
            run_data.final_agent_states.append(agent_state)

        # Collect events from database
        run_data.events = self._query_events(run_id)

        # Compute metrics
        run_data.metrics = self._compute_run_metrics(run_data)

        return run_data

    def _query_events(self, run_id: str) -> List[Dict[str, Any]]:
        """Query simulation events from database."""
        events = []
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT event_type, agent_id_primary, agent_id_secondary,
                       step_timestamp, description, details
                FROM simulation_events
                WHERE simulation_run_id = ?
                ORDER BY step_timestamp
            """, (run_id,))

            for row in cursor.fetchall():
                events.append({
                    "event_type": row[0],
                    "agent_primary": row[1],
                    "agent_secondary": row[2],
                    "step": row[3],
                    "description": row[4],
                    "details": json.loads(row[5]) if row[5] else {},
                })

            conn.close()
        except Exception as e:
            logger.warning(f"Could not query events: {e}")

        return events

    def _compute_run_metrics(self, run_data: RunData) -> Dict[str, List[float]]:
        """Compute metrics from run data."""
        metrics = {}

        agents = run_data.final_agent_states

        if not agents:
            return metrics

        # Social metrics
        metrics["avg_happiness"] = [statistics.mean([a["happiness"] for a in agents])]
        metrics["avg_energy"] = [statistics.mean([a["energy"] for a in agents])]
        metrics["avg_health"] = [statistics.mean([a["health"] for a in agents])]

        # Network metrics
        connection_counts = [a["num_connections"] for a in agents]
        metrics["avg_connections"] = [statistics.mean(connection_counts)]
        metrics["max_connections"] = [max(connection_counts)]
        metrics["isolated_agents"] = [sum(1 for c in connection_counts if c == 0)]

        # Compute network density
        n = len(agents)
        total_possible = n * (n - 1)
        total_connections = sum(connection_counts)
        metrics["network_density"] = [total_connections / total_possible if total_possible > 0 else 0]

        # Economic metrics
        wealth_values = [
            a["resources"].get("currency", 0) + sum(a["resources"].values())
            for a in agents
        ]
        metrics["avg_wealth"] = [statistics.mean(wealth_values)]
        metrics["wealth_std"] = [statistics.stdev(wealth_values) if len(wealth_values) > 1 else 0]

        # Gini coefficient
        metrics["wealth_gini"] = [self._compute_gini(wealth_values)]

        # Event counts
        event_counts = {}
        for event in run_data.events:
            event_type = event["event_type"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        metrics["total_events"] = [len(run_data.events)]
        metrics["action_events"] = [event_counts.get("AGENT_ACTION_CHOSEN", 0)]
        metrics["social_events"] = [event_counts.get("SOCIAL_INTERACTION", 0)]
        metrics["trade_events"] = [event_counts.get("TRADE_COMPLETED", 0)]

        # Build action type distribution from AGENT_ACTION_CHOSEN events
        action_distribution = {}
        llm_count = 0
        fallback_count = 0
        for event in run_data.events:
            if event["event_type"] == "AGENT_ACTION_CHOSEN":
                details = event.get("details", {})
                action_type = details.get("action_type", "unknown")
                action_distribution[action_type] = action_distribution.get(action_type, 0) + 1
                # Track source attribution
                source = details.get("source", "unknown")
                if source == "llm":
                    llm_count += 1
                elif source == "fallback":
                    fallback_count += 1

        # Store action distribution as metrics
        for action_type, count in action_distribution.items():
            metrics[f"action_{action_type}"] = [count]

        metrics["llm_decisions"] = [llm_count]
        metrics["fallback_decisions"] = [fallback_count]
        total_decisions = llm_count + fallback_count
        metrics["llm_decision_rate"] = [llm_count / total_decisions if total_decisions > 0 else 0]

        # Cultural metrics
        cultural_groups = [a["cultural_group"] for a in agents if a["cultural_group"] is not None]
        if cultural_groups:
            unique_groups = len(set(cultural_groups))
            metrics["cultural_diversity"] = [unique_groups / len(cultural_groups)]

        return metrics

    def _compute_gini(self, values: List[float]) -> float:
        """Compute Gini coefficient for inequality measurement."""
        if not values or len(values) < 2:
            return 0.0

        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = sum((i + 1) * v for i, v in enumerate(sorted_values))
        return (2 * cumsum) / (n * sum(sorted_values)) - (n + 1) / n if sum(sorted_values) > 0 else 0

    def _analyze_results(
        self,
        experiment_id: str,
        experiment: Experiment,
        start_time: datetime,
        end_time: datetime,
        run_ids: List[str],
        num_agents: int,
        num_steps: int,
    ) -> ExperimentResult:
        """Analyze collected data and generate insights."""

        # Aggregate metrics across runs
        aggregated_metrics: Dict[str, List[float]] = {}
        for run_data in self.run_data:
            for metric_name, values in run_data.metrics.items():
                if metric_name not in aggregated_metrics:
                    aggregated_metrics[metric_name] = []
                aggregated_metrics[metric_name].extend(values)

        # Compute summary statistics
        summary_stats = {}
        for metric_name, values in aggregated_metrics.items():
            if values:
                summary_stats[metric_name] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "n": len(values),
                }

        # Aggregate event counts
        total_event_counts: Dict[str, int] = {}
        for run_data in self.run_data:
            for event in run_data.events:
                event_type = event["event_type"]
                total_event_counts[event_type] = total_event_counts.get(event_type, 0) + 1

        # Collect agent trajectories
        agent_trajectories = []
        for run_data in self.run_data:
            for agent_state in run_data.final_agent_states:
                agent_trajectories.append({
                    "run_id": run_data.run_id,
                    **agent_state,
                })

        # Compute correlations
        correlations = self._compute_correlations(agent_trajectories)

        # Generate findings
        findings = self._generate_findings(
            experiment=experiment,
            summary_stats=summary_stats,
            event_counts=total_event_counts,
            correlations=correlations,
        )

        # Generate conclusion
        conclusion = self._generate_conclusion(experiment, findings, summary_stats)

        return ExperimentResult(
            experiment_id=experiment_id,
            experiment_type=experiment.name,
            hypothesis=experiment.hypothesis,
            start_time=start_time,
            end_time=end_time,
            config_used=experiment.base_config_overrides,
            raw_metrics=aggregated_metrics,
            event_counts=total_event_counts,
            agent_trajectories=agent_trajectories,
            summary_statistics=summary_stats,
            correlations=correlations,
            findings=findings,
            conclusion=conclusion,
            num_agents=num_agents,
            num_steps=num_steps,
            run_ids=run_ids,
        )

    def _compute_correlations(self, trajectories: List[Dict]) -> Dict[str, float]:
        """Compute correlations between key variables."""
        correlations = {}

        if len(trajectories) < 3:
            return correlations

        def pearson_correlation(x: List[float], y: List[float]) -> float:
            if len(x) != len(y) or len(x) < 2:
                return 0.0
            n = len(x)
            mean_x, mean_y = statistics.mean(x), statistics.mean(y)
            cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
            std_x = statistics.stdev(x) if len(x) > 1 else 1
            std_y = statistics.stdev(y) if len(y) > 1 else 1
            return cov / (std_x * std_y) if std_x > 0 and std_y > 0 else 0

        # Extract variables
        happiness = [t["happiness"] for t in trajectories]
        connections = [t["num_connections"] for t in trajectories]
        wealth = [sum(t["resources"].values()) for t in trajectories]
        health = [t["health"] for t in trajectories]
        reputation = [t["social_reputation"] for t in trajectories]

        # Compute correlations
        correlations["happiness_connections"] = pearson_correlation(happiness, connections)
        correlations["happiness_wealth"] = pearson_correlation(happiness, wealth)
        correlations["happiness_health"] = pearson_correlation(happiness, health)
        correlations["connections_reputation"] = pearson_correlation(connections, reputation)
        correlations["wealth_reputation"] = pearson_correlation(wealth, reputation)

        return correlations

    def _generate_findings(
        self,
        experiment: Experiment,
        summary_stats: Dict[str, Dict[str, float]],
        event_counts: Dict[str, int],
        correlations: Dict[str, float],
    ) -> List[str]:
        """Generate human-readable findings from the data."""
        findings = []

        # Social findings
        if "avg_connections" in summary_stats:
            avg_conn = summary_stats["avg_connections"]["mean"]
            findings.append(
                f"Agents formed an average of {avg_conn:.2f} social connections"
            )

        if "network_density" in summary_stats:
            density = summary_stats["network_density"]["mean"]
            findings.append(
                f"Network density was {density:.3f} (0=no connections, 1=fully connected)"
            )

        if "isolated_agents" in summary_stats:
            isolated = summary_stats["isolated_agents"]["mean"]
            findings.append(f"Average of {isolated:.1f} agents remained isolated")

        # Wellbeing findings
        if "avg_happiness" in summary_stats:
            happiness = summary_stats["avg_happiness"]["mean"]
            findings.append(f"Mean agent happiness was {happiness:.3f} (0-1 scale)")

        # Economic findings
        if "wealth_gini" in summary_stats:
            gini = summary_stats["wealth_gini"]["mean"]
            inequality = "low" if gini < 0.3 else "moderate" if gini < 0.5 else "high"
            findings.append(f"Wealth inequality was {inequality} (Gini={gini:.3f})")

        # Correlation findings
        for corr_name, corr_value in correlations.items():
            if abs(corr_value) > 0.3:
                direction = "positive" if corr_value > 0 else "negative"
                strength = "strong" if abs(corr_value) > 0.6 else "moderate"
                var1, var2 = corr_name.replace("_", " ").split(" ", 1)
                findings.append(
                    f"Found {strength} {direction} correlation between {var1} and {var2} (r={corr_value:.3f})"
                )

        # Activity findings
        total_events = sum(event_counts.values())
        findings.append(f"Total of {total_events} events recorded across all runs")

        return findings

    def _generate_conclusion(
        self,
        experiment: Experiment,
        findings: List[str],
        summary_stats: Dict[str, Dict[str, float]],
    ) -> str:
        """Generate a conclusion about the experiment hypothesis."""

        conclusion_parts = [f"Experiment: {experiment.name}"]
        conclusion_parts.append(f"Hypothesis: {experiment.hypothesis}")
        conclusion_parts.append("")
        conclusion_parts.append("Key Findings:")
        for i, finding in enumerate(findings[:5], 1):
            conclusion_parts.append(f"  {i}. {finding}")

        conclusion_parts.append("")

        # Evaluate hypothesis support
        if experiment.name == "social_dynamics":
            if "avg_connections" in summary_stats:
                avg_conn = summary_stats["avg_connections"]["mean"]
                if avg_conn > 1:
                    conclusion_parts.append(
                        "CONCLUSION: The hypothesis is SUPPORTED. Agents formed social "
                        "connections, indicating emergent social network formation."
                    )
                else:
                    conclusion_parts.append(
                        "CONCLUSION: The hypothesis is PARTIALLY SUPPORTED. Limited social "
                        "connections formed, possibly due to short simulation duration."
                    )
        else:
            conclusion_parts.append(
                "CONCLUSION: Further analysis needed to fully evaluate hypothesis."
            )

        return "\n".join(conclusion_parts)

    def quick_experiment(
        self,
        num_agents: int = 5,
        num_steps: int = 20,
    ) -> Dict[str, Any]:
        """
        Run a quick experiment and return summary data.

        This is a synchronous wrapper for simple use cases.
        """
        return asyncio.run(
            self.run_experiment(
                experiment_type="social_dynamics",
                num_agents=num_agents,
                num_steps=num_steps,
                num_runs=1,
            )
        )
