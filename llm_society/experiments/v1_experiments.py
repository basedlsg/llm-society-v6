"""
V1.0 Baseline Experiments for LLM Society

Implements the three baseline experiments:
1. Single-Model Behavioral Fingerprint
2. Model Comparison (A vs B)
3. Scarcity vs Abundance (Environmental Comparison)
"""

import asyncio
import json
import logging
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import numpy as np

from ..metrics import (
    HappinessCalculator,
    MetricsCollector,
    BehavioralFingerprint,
    compute_behavioral_fingerprint,
)
from ..metrics.happiness import calculate_population_happiness

logger = logging.getLogger(__name__)

# Default experiment seeds for reproducibility
DEFAULT_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

# V1.0 World defaults
# NOTE: World size reduced from 100x100 to 20x20 to ensure agents can interact.
# With 50 agents in 20x20 grid (400 units): avg spacing ~2.8 units
# Social radius 10.0 means most agents can reach each other within a few steps.
# For sparser experiments, scale world_size up but ensure agents can meet.
V1_WORLD_CONFIG = {
    "world_size": (20.0, 20.0),  # Smaller world = denser agent population
    "social_radius": 10.0,
    "trade_radius": 10.0,
    "interaction_radius": 5.0,
    "energy_decay_per_step": 0.005,  # Energy decay to create survival pressure
    "food_consumption_interval": 20,  # Consume food every 20 steps
}

# V1.0 Initial resources
V1_INITIAL_RESOURCES = {
    "food": 10,
    "currency": 500,
    "materials": 5,
    "energy_item": 5,
}

# Scarcity/Abundance variants
SCARCITY_RESOURCES = {
    "food": 5,
    "currency": 200,
    "materials": 2,
    "energy_item": 2,
}

ABUNDANCE_RESOURCES = {
    "food": 20,
    "currency": 1000,
    "materials": 10,
    "energy_item": 10,
}


@dataclass
class ExperimentRunResult:
    """Result from a single simulation run."""
    run_id: str
    seed: int
    model_name: str
    condition: str
    n_agents: int
    max_steps: int

    # Fingerprint
    fingerprint: BehavioralFingerprint

    # Raw metrics
    happiness_trajectory: List[Dict[str, float]]
    final_agent_states: List[Dict[str, Any]]

    # Timing
    start_time: datetime
    end_time: datetime
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "seed": self.seed,
            "model_name": self.model_name,
            "condition": self.condition,
            "n_agents": self.n_agents,
            "max_steps": self.max_steps,
            "fingerprint": self.fingerprint.to_dict(),
            "happiness_trajectory": self.happiness_trajectory,
            "final_agent_states": self.final_agent_states,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class AggregatedResults:
    """Aggregated results across multiple runs."""
    experiment_name: str
    model_name: str
    condition: str
    n_runs: int

    # Mean ± Std for each metric
    metrics: Dict[str, Dict[str, float]]

    # Individual run results
    run_results: List[ExperimentRunResult]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "condition": self.condition,
            "n_runs": self.n_runs,
            "metrics": self.metrics,
            "run_results": [r.to_dict() for r in self.run_results],
        }


@dataclass
class ComparisonResult:
    """Result of comparing two conditions or models."""
    metric_name: str
    value_a: float
    value_b: float
    difference: float
    pct_change: float
    p_value: float
    test_used: str
    cohens_d: float
    significant: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseV1Experiment:
    """Base class for v1.0 experiments."""

    def __init__(
        self,
        n_agents: int = 50,
        max_steps: int = 1000,
        seeds: Optional[List[int]] = None,
        output_dir: str = "./experiment_results",
    ):
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.seeds = seeds or DEFAULT_SEEDS[:10]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def run_single(
        self,
        seed: int,
        model_name: str,
        condition: str,
        initial_resources: Dict[str, int],
    ) -> ExperimentRunResult:
        """Run a single simulation with given parameters."""
        from ..simulation.society_simulator import SocietySimulator
        from ..utils.config import Config

        start_time = datetime.now()

        # Create config
        config = Config.default()
        config.agents.count = self.n_agents
        config.simulation.max_steps = self.max_steps
        config.simulation.seed = seed
        config.llm.model_name = model_name

        # Apply v1.0 world config
        config.simulation.world_size = V1_WORLD_CONFIG["world_size"]
        config.agents.social_radius = V1_WORLD_CONFIG["social_radius"]

        # Create metrics collector
        metrics_collector = MetricsCollector()
        run_id = f"{condition}_{model_name}_{seed}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        metrics_collector.start_run(run_id, config.to_dict())

        # Run simulation
        simulator = SocietySimulator(config)

        # Override initial resources
        for agent in simulator._agent_list:
            agent.resources = initial_resources.copy()

        try:
            # Run with metrics collection
            for step in range(self.max_steps):
                await simulator._async_step()
                metrics_collector.collect_step(step, simulator._agent_list)

                if step % 100 == 0:
                    logger.info(f"Run {run_id}: Step {step}/{self.max_steps}")

        except Exception as e:
            logger.error(f"Run {run_id} failed: {e}")
            raise
        finally:
            if hasattr(simulator, 'llm_coordinator') and simulator.llm_coordinator:
                await simulator.llm_coordinator.stop()

        metrics_collector.end_run()
        end_time = datetime.now()

        # Compute fingerprint
        fingerprint = metrics_collector.generate_fingerprint(simulator._agent_list)

        # Collect final agent states
        final_states = []
        for agent in simulator._agent_list:
            final_states.append({
                "agent_id": agent.unique_id,
                "position": {"x": agent.position.x, "y": agent.position.y},
                "energy": agent.energy,
                "health": agent.health,
                "resources": dict(agent.resources),
                "social_connections": dict(agent.social_connections),
                "num_connections": len(agent.social_connections),
            })

        return ExperimentRunResult(
            run_id=run_id,
            seed=seed,
            model_name=model_name,
            condition=condition,
            n_agents=self.n_agents,
            max_steps=self.max_steps,
            fingerprint=fingerprint,
            happiness_trajectory=metrics_collector.get_happiness_trajectory(),
            final_agent_states=final_states,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=(end_time - start_time).total_seconds(),
        )

    def aggregate_results(
        self,
        results: List[ExperimentRunResult],
        experiment_name: str,
    ) -> AggregatedResults:
        """Aggregate results from multiple runs."""
        if not results:
            raise ValueError("No results to aggregate")

        # Get model and condition from first result
        model_name = results[0].model_name
        condition = results[0].condition

        # Extract metric values
        metric_values: Dict[str, List[float]] = {}

        fingerprint_fields = [
            "mean_degree", "isolation_fraction", "clustering_coefficient",
            "gini_currency", "mean_food", "starvation_rate", "trade_frequency",
            "survival_rate", "rest_fraction", "move_fraction", "talk_fraction",
            "gather_fraction", "trade_fraction", "mean_happiness", "happiness_std",
        ]

        for field_name in fingerprint_fields:
            values = [getattr(r.fingerprint, field_name) for r in results]
            values = [v for v in values if v is not None]
            metric_values[field_name] = values

        # Compute statistics
        metrics = {}
        for metric_name, values in metric_values.items():
            if values:
                metrics[metric_name] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "n": len(values),
                }

        return AggregatedResults(
            experiment_name=experiment_name,
            model_name=model_name,
            condition=condition,
            n_runs=len(results),
            metrics=metrics,
            run_results=results,
        )

    def compare_conditions(
        self,
        results_a: AggregatedResults,
        results_b: AggregatedResults,
    ) -> List[ComparisonResult]:
        """Compare two sets of results statistically."""
        comparisons = []

        # Get metric names that exist in both
        metric_names = set(results_a.metrics.keys()) & set(results_b.metrics.keys())

        for metric_name in metric_names:
            # Get values from individual runs
            values_a = [getattr(r.fingerprint, metric_name) for r in results_a.run_results]
            values_b = [getattr(r.fingerprint, metric_name) for r in results_b.run_results]

            values_a = [v for v in values_a if v is not None]
            values_b = [v for v in values_b if v is not None]

            if len(values_a) < 2 or len(values_b) < 2:
                continue

            # Normality test
            _, p_norm_a = stats.shapiro(values_a)
            _, p_norm_b = stats.shapiro(values_b)

            if p_norm_a > 0.05 and p_norm_b > 0.05:
                # Normal: use t-test
                stat, p_value = stats.ttest_ind(values_a, values_b)
                test_used = "t-test"
            else:
                # Non-normal: use Mann-Whitney
                stat, p_value = stats.mannwhitneyu(values_a, values_b, alternative='two-sided')
                test_used = "Mann-Whitney"

            # Effect size (Cohen's d)
            mean_a = np.mean(values_a)
            mean_b = np.mean(values_b)
            mean_diff = mean_b - mean_a
            pooled_std = np.sqrt((np.std(values_a)**2 + np.std(values_b)**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

            # Percentage change
            pct_change = (mean_diff / mean_a * 100) if mean_a != 0 else float('inf')

            comparisons.append(ComparisonResult(
                metric_name=metric_name,
                value_a=mean_a,
                value_b=mean_b,
                difference=mean_diff,
                pct_change=pct_change,
                p_value=p_value,
                test_used=test_used,
                cohens_d=cohens_d,
                significant=p_value < 0.05,
            ))

        return comparisons


class BaselineFingerprint(BaseV1Experiment):
    """
    Experiment 1: Single-Model Behavioral Fingerprint

    Purpose: Get baseline behavioral fingerprint for a single model.
    """

    def __init__(
        self,
        model_name: str = "gemini-pro",
        n_agents: int = 50,
        max_steps: int = 1000,
        seeds: Optional[List[int]] = None,
        output_dir: str = "./experiment_results/exp1_baseline",
    ):
        super().__init__(n_agents, max_steps, seeds, output_dir)
        self.model_name = model_name

    async def run(self) -> AggregatedResults:
        """Run the baseline fingerprint experiment."""
        logger.info(f"Starting Baseline Fingerprint experiment for {self.model_name}")
        logger.info(f"Runs: {len(self.seeds)}, Agents: {self.n_agents}, Steps: {self.max_steps}")

        results = []

        for i, seed in enumerate(self.seeds):
            logger.info(f"Running seed {seed} ({i+1}/{len(self.seeds)})")

            result = await self.run_single(
                seed=seed,
                model_name=self.model_name,
                condition="baseline",
                initial_resources=V1_INITIAL_RESOURCES,
            )
            results.append(result)

            # Save individual result
            result_path = self.output_dir / f"run_seed_{seed}.json"
            with open(result_path, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)

        # Aggregate
        aggregated = self.aggregate_results(results, "baseline_fingerprint")

        # Save aggregated
        agg_path = self.output_dir / "aggregated_results.json"
        with open(agg_path, 'w') as f:
            json.dump(aggregated.to_dict(), f, indent=2)

        # Generate report
        self._generate_report(aggregated)

        logger.info(f"Baseline Fingerprint experiment completed")
        return aggregated

    def _generate_report(self, results: AggregatedResults):
        """Generate markdown report."""
        report = f"""# Behavioral Fingerprint: {results.model_name}

## Configuration
- Model: {results.model_name}
- Agents: {self.n_agents}
- Steps: {self.max_steps}
- Runs: {results.n_runs}

## Summary Statistics

### Network Metrics
| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
"""
        for metric in ["mean_degree", "isolation_fraction", "clustering_coefficient"]:
            if metric in results.metrics:
                m = results.metrics[metric]
                report += f"| {metric} | {m['mean']:.3f} | {m['std']:.3f} | {m['min']:.3f} | {m['max']:.3f} |\n"

        report += """
### Economic Metrics
| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
"""
        for metric in ["gini_currency", "mean_food", "starvation_rate"]:
            if metric in results.metrics:
                m = results.metrics[metric]
                report += f"| {metric} | {m['mean']:.3f} | {m['std']:.3f} | {m['min']:.3f} | {m['max']:.3f} |\n"

        report += """
### Activity Profile
| Action | Mean % | Std % |
|--------|--------|-------|
"""
        for metric in ["rest_fraction", "move_fraction", "talk_fraction", "gather_fraction", "trade_fraction"]:
            if metric in results.metrics:
                m = results.metrics[metric]
                report += f"| {metric.replace('_fraction', '')} | {m['mean']*100:.1f}% | {m['std']*100:.1f}% |\n"

        report += """
### Happiness (Analysis Only)
| Metric | Value |
|--------|-------|
"""
        if "mean_happiness" in results.metrics:
            report += f"| Mean | {results.metrics['mean_happiness']['mean']:.3f} |\n"
        if "happiness_std" in results.metrics:
            report += f"| Std | {results.metrics['happiness_std']['mean']:.3f} |\n"

        report_path = self.output_dir / "summary_report.md"
        with open(report_path, 'w') as f:
            f.write(report)


class ModelComparison(BaseV1Experiment):
    """
    Experiment 2: Model Comparison (A vs B)

    Purpose: Compare two models under identical conditions.
    """

    def __init__(
        self,
        models: List[str],
        n_agents: int = 50,
        max_steps: int = 1000,
        seeds: Optional[List[int]] = None,
        output_dir: str = "./experiment_results/exp2_model_comparison",
    ):
        super().__init__(n_agents, max_steps, seeds, output_dir)
        self.models = models

    async def run(self) -> Tuple[Dict[str, AggregatedResults], List[ComparisonResult]]:
        """Run the model comparison experiment."""
        logger.info(f"Starting Model Comparison experiment: {self.models}")

        all_results = {}

        for model_name in self.models:
            logger.info(f"Running model: {model_name}")
            model_dir = self.output_dir / model_name
            model_dir.mkdir(parents=True, exist_ok=True)

            results = []
            for i, seed in enumerate(self.seeds):
                logger.info(f"Model {model_name}, seed {seed} ({i+1}/{len(self.seeds)})")

                result = await self.run_single(
                    seed=seed,
                    model_name=model_name,
                    condition="baseline",
                    initial_resources=V1_INITIAL_RESOURCES,
                )
                results.append(result)

                # Save individual result
                result_path = model_dir / f"run_seed_{seed}.json"
                with open(result_path, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)

            # Aggregate for this model
            aggregated = self.aggregate_results(results, f"model_comparison_{model_name}")
            all_results[model_name] = aggregated

            # Save aggregated
            agg_path = model_dir / "aggregated_results.json"
            with open(agg_path, 'w') as f:
                json.dump(aggregated.to_dict(), f, indent=2)

        # Compare models
        comparisons = []
        if len(self.models) == 2:
            comparisons = self.compare_conditions(
                all_results[self.models[0]],
                all_results[self.models[1]],
            )

            # Save comparisons
            comp_path = self.output_dir / "comparison_results.json"
            with open(comp_path, 'w') as f:
                json.dump([c.to_dict() for c in comparisons], f, indent=2)

        # Generate report
        self._generate_report(all_results, comparisons)

        return all_results, comparisons

    def _generate_report(
        self,
        all_results: Dict[str, AggregatedResults],
        comparisons: List[ComparisonResult],
    ):
        """Generate comparison report."""
        report = f"""# Model Comparison: {' vs '.join(self.models)}

## Configuration
- Models: {', '.join(self.models)}
- Agents: {self.n_agents}
- Steps: {self.max_steps}
- Runs per model: {len(self.seeds)}

## Key Differences

| Metric | {self.models[0]} | {self.models[1] if len(self.models) > 1 else 'N/A'} | Diff | p-value | Sig |
|--------|---------|---------|------|---------|-----|
"""
        for comp in comparisons:
            sig = "✓" if comp.significant else ""
            report += f"| {comp.metric_name} | {comp.value_a:.3f} | {comp.value_b:.3f} | {comp.difference:+.3f} | {comp.p_value:.4f} | {sig} |\n"

        report_path = self.output_dir / "comparison_report.md"
        with open(report_path, 'w') as f:
            f.write(report)


class EnvironmentalComparison(BaseV1Experiment):
    """
    Experiment 3: Scarcity vs Abundance

    Purpose: Test how same model behaves under different resource regimes.
    """

    def __init__(
        self,
        model_name: str = "gemini-pro",
        n_agents: int = 50,
        max_steps: int = 1000,
        seeds: Optional[List[int]] = None,
        output_dir: str = "./experiment_results/exp3_environmental",
    ):
        super().__init__(n_agents, max_steps, seeds, output_dir)
        self.model_name = model_name

    async def run(self) -> Tuple[Dict[str, AggregatedResults], List[ComparisonResult]]:
        """Run the environmental comparison experiment."""
        logger.info(f"Starting Environmental Comparison experiment")

        conditions = {
            "scarcity": SCARCITY_RESOURCES,
            "abundance": ABUNDANCE_RESOURCES,
        }

        all_results = {}

        for condition_name, resources in conditions.items():
            logger.info(f"Running condition: {condition_name}")
            condition_dir = self.output_dir / condition_name
            condition_dir.mkdir(parents=True, exist_ok=True)

            results = []
            for i, seed in enumerate(self.seeds):
                logger.info(f"Condition {condition_name}, seed {seed} ({i+1}/{len(self.seeds)})")

                result = await self.run_single(
                    seed=seed,
                    model_name=self.model_name,
                    condition=condition_name,
                    initial_resources=resources,
                )
                results.append(result)

                # Save individual result
                result_path = condition_dir / f"run_seed_{seed}.json"
                with open(result_path, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)

            # Aggregate
            aggregated = self.aggregate_results(results, f"environmental_{condition_name}")
            all_results[condition_name] = aggregated

            # Save aggregated
            agg_path = condition_dir / "aggregated_results.json"
            with open(agg_path, 'w') as f:
                json.dump(aggregated.to_dict(), f, indent=2)

        # Compare conditions
        comparisons = self.compare_conditions(
            all_results["scarcity"],
            all_results["abundance"],
        )

        # Save comparisons
        comp_path = self.output_dir / "condition_comparison.json"
        with open(comp_path, 'w') as f:
            json.dump([c.to_dict() for c in comparisons], f, indent=2)

        # Generate report
        self._generate_report(all_results, comparisons)

        return all_results, comparisons

    def _generate_report(
        self,
        all_results: Dict[str, AggregatedResults],
        comparisons: List[ComparisonResult],
    ):
        """Generate environmental effects report."""
        report = f"""# Environmental Effects: Scarcity vs Abundance

## Configuration
- Model: {self.model_name}
- Agents: {self.n_agents}
- Steps: {self.max_steps}
- Runs per condition: {len(self.seeds)}

## Conditions
| Resource | Scarcity | Abundance | Ratio |
|----------|----------|-----------|-------|
| Food | {SCARCITY_RESOURCES['food']} | {ABUNDANCE_RESOURCES['food']} | {ABUNDANCE_RESOURCES['food']/SCARCITY_RESOURCES['food']:.1f}x |
| Currency | {SCARCITY_RESOURCES['currency']} | {ABUNDANCE_RESOURCES['currency']} | {ABUNDANCE_RESOURCES['currency']/SCARCITY_RESOURCES['currency']:.1f}x |
| Materials | {SCARCITY_RESOURCES['materials']} | {ABUNDANCE_RESOURCES['materials']} | {ABUNDANCE_RESOURCES['materials']/SCARCITY_RESOURCES['materials']:.1f}x |

## Behavioral Changes

| Metric | Scarcity | Abundance | Δ | Direction |
|--------|----------|-----------|---|-----------|
"""
        for comp in comparisons:
            direction = "↑" if comp.difference > 0 else "↓" if comp.difference < 0 else "="
            report += f"| {comp.metric_name} | {comp.value_a:.3f} | {comp.value_b:.3f} | {comp.difference:+.3f} | {direction} |\n"

        # Hypothesis testing
        report += """
## Hypothesis Results
| Hypothesis | Supported? | Evidence |
|------------|------------|----------|
"""
        # Check each hypothesis
        hypotheses = [
            ("H1: More trading under scarcity", "trade_fraction", "<"),
            ("H2: More gathering under scarcity", "gather_fraction", "<"),
            ("H3: Lower survival under scarcity", "survival_rate", "<"),
            ("H4: More social under abundance", "mean_degree", ">"),
            ("H5: Higher inequality under scarcity", "gini_currency", "<"),
        ]

        for hyp_name, metric, expected_dir in hypotheses:
            comp = next((c for c in comparisons if c.metric_name == metric), None)
            if comp:
                if expected_dir == "<":
                    supported = comp.value_a > comp.value_b
                else:
                    supported = comp.value_a < comp.value_b
                status = "Yes" if supported and comp.significant else "No"
                evidence = f"Δ={comp.difference:+.3f}, p={comp.p_value:.4f}"
                report += f"| {hyp_name} | {status} | {evidence} |\n"

        report_path = self.output_dir / "environmental_effects_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
