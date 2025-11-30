#!/usr/bin/env python3
"""
Main entry point for 2,500-Agent LLM Society Simulation
"""

import asyncio
import logging
import os
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

# Add llm_society to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_society.monitoring.metrics import MetricsCollector
from llm_society.simulation.society_simulator import SocietySimulator
from llm_society.utils.config import Config

app = typer.Typer(name="llm-society", help="2,500-Agent LLM Society Simulation")
console = Console()


def setup_logging(debug: bool = False):
    """Set up rich logging"""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def run(
    agents: int = typer.Option(
        50, "--agents", "-a", help="Number of agents to simulate"
    ),
    steps: int = typer.Option(1000, "--steps", "-s", help="Number of simulation steps"),
    model: str = typer.Option("gemini-pro", "--model", "-m", help="LLM model to use"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    output_dir: str = typer.Option(
        "./results", "--output", "-o", help="Output directory for results"
    ),
):
    """Run the LLM society simulation"""
    setup_logging(debug)
    logger = logging.getLogger(__name__)

    console.print(
        "🚀 [bold green]Starting 2,500-Agent LLM Society Simulation[/bold green]"
    )
    console.print(f"📊 Agents: {agents}")
    console.print(f"⏱️  Steps: {steps}")
    console.print(f"🤖 Model: {model}")

    try:
        # Load configuration
        config = Config.load(config_file) if config_file else Config.default()
        config.agents.count = agents
        config.simulation.max_steps = steps
        config.llm.model_name = model
        config.output.directory = output_dir

        # Create simulator
        simulator = SocietySimulator(config)

        # Run simulation
        asyncio.run(simulator.run())

        console.print("✅ [bold green]Simulation completed successfully![/bold green]")

    except KeyboardInterrupt:
        console.print("⚠️  [yellow]Simulation interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"❌ [bold red]Simulation failed: {e}[/bold red]")
        if debug:
            console.print_exception()
        sys.exit(1)


@app.command()
def demo(
    scenario: str = typer.Option(
        "basic", "--scenario", "-s", help="Demo scenario to run"
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
):
    """Run a demo scenario"""
    setup_logging(debug)
    console.print(f"🎭 [bold blue]Running demo scenario: {scenario}[/bold blue]")

    from llm_society.simulation.demo_scenarios import run_demo_scenario

    asyncio.run(run_demo_scenario(scenario))


@app.command()
def benchmark(
    agents: int = typer.Option(
        100, "--agents", "-a", help="Number of agents for benchmark"
    ),
    duration: int = typer.Option(
        60, "--duration", "-d", help="Benchmark duration in seconds"
    ),
):
    """Run performance benchmarks"""
    console.print("⚡ [bold yellow]Running performance benchmark[/bold yellow]")
    console.print(f"📊 Agents: {agents}")
    console.print(f"⏱️  Duration: {duration}s")

    from llm_society.monitoring.benchmarks import run_benchmark

    asyncio.run(run_benchmark(agents, duration))


@app.command()
def install_deps():
    """Install additional dependencies"""
    console.print("📦 [bold blue]Installing additional dependencies...[/bold blue]")
    os.system("./install_dev_dependencies.sh")


@app.command()
def experiment(
    experiment_type: str = typer.Option(
        "social_dynamics",
        "--type",
        "-t",
        help="Experiment type: social_dynamics, economic_behavior, emergent_culture, cooperation_dynamics",
    ),
    agents: int = typer.Option(5, "--agents", "-a", help="Number of agents per run"),
    steps: int = typer.Option(20, "--steps", "-s", help="Simulation steps per run"),
    runs: int = typer.Option(1, "--runs", "-r", help="Number of replications"),
    model: str = typer.Option("gemini-pro", "--model", "-m", help="LLM model to use"),
    output_dir: str = typer.Option(
        "./experiment_results", "--output", "-o", help="Output directory for results"
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
):
    """Run a controlled experiment on the LLM society simulation"""
    setup_logging(debug)
    logger = logging.getLogger(__name__)

    console.print("🔬 [bold cyan]Starting LLM Society Experiment[/bold cyan]")
    console.print(f"📋 Experiment Type: {experiment_type}")
    console.print(f"📊 Agents: {agents}")
    console.print(f"⏱️  Steps: {steps}")
    console.print(f"🔄 Runs: {runs}")
    console.print(f"🤖 Model: {model}")

    try:
        from llm_society.experiments import ExperimentCoordinator

        coordinator = ExperimentCoordinator(output_dir=output_dir)

        # List available experiments
        console.print(f"\n📑 Available experiments: {coordinator.list_experiments()}")

        # Run the experiment
        result = asyncio.run(
            coordinator.run_experiment(
                experiment_type=experiment_type,
                num_agents=agents,
                num_steps=steps,
                num_runs=runs,
                model_name=model,
            )
        )

        # Display results
        console.print("\n" + "=" * 60)
        console.print("[bold green]EXPERIMENT RESULTS[/bold green]")
        console.print("=" * 60)
        console.print(f"\n[bold]Experiment:[/bold] {result.experiment_type}")
        console.print(f"[bold]Hypothesis:[/bold] {result.hypothesis}")
        console.print(f"[bold]Duration:[/bold] {result.end_time - result.start_time}")

        console.print("\n[bold]Summary Statistics:[/bold]")
        for metric, stats in result.summary_statistics.items():
            console.print(f"  • {metric}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

        console.print("\n[bold]Key Findings:[/bold]")
        for i, finding in enumerate(result.findings, 1):
            console.print(f"  {i}. {finding}")

        console.print("\n[bold]Conclusion:[/bold]")
        console.print(result.conclusion)

        console.print(f"\n📁 Full results saved to: {output_dir}/{result.experiment_id}_results.json")
        console.print("\n✅ [bold green]Experiment completed successfully![/bold green]")

    except Exception as e:
        console.print(f"❌ [bold red]Experiment failed: {e}[/bold red]")
        if debug:
            console.print_exception()
        sys.exit(1)


@app.command()
def experiment_v1(
    exp_type: str = typer.Option(
        "baseline",
        "--type",
        "-t",
        help="V1.0 experiment type: baseline, model_comparison, environmental",
    ),
    agents: int = typer.Option(50, "--agents", "-a", help="Number of agents per run"),
    steps: int = typer.Option(1000, "--steps", "-s", help="Simulation steps per run"),
    runs: int = typer.Option(10, "--runs", "-r", help="Number of replications"),
    model: str = typer.Option("gemini-pro", "--model", "-m", help="Primary LLM model"),
    model_b: str = typer.Option(
        None, "--model-b", help="Secondary model (for model_comparison)"
    ),
    output_dir: str = typer.Option(
        "./experiment_results", "--output", "-o", help="Output directory"
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
):
    """Run v1.0 baseline experiments (fingerprint, model comparison, environmental)"""
    setup_logging(debug)

    console.print("[bold cyan]V1.0 Baseline Experiment[/bold cyan]")
    console.print(f"Type: {exp_type}")
    console.print(f"Agents: {agents}, Steps: {steps}, Runs: {runs}")

    try:
        from llm_society.experiments import (
            BaselineFingerprint,
            ModelComparison,
            EnvironmentalComparison,
        )

        if exp_type == "baseline":
            console.print(f"\n[bold]Running Baseline Fingerprint for {model}[/bold]")
            exp = BaselineFingerprint(
                model_name=model,
                n_agents=agents,
                max_steps=steps,
                seeds=list(range(runs)),
                output_dir=f"{output_dir}/exp1_baseline",
            )
            result = asyncio.run(exp.run())

            console.print("\n[bold green]Results:[/bold green]")
            for metric, stats in result.metrics.items():
                console.print(f"  {metric}: {stats['mean']:.3f} +/- {stats['std']:.3f}")

        elif exp_type == "model_comparison":
            if not model_b:
                console.print("[red]Error: --model-b required for model_comparison[/red]")
                sys.exit(1)

            console.print(f"\n[bold]Comparing {model} vs {model_b}[/bold]")
            exp = ModelComparison(
                models=[model, model_b],
                n_agents=agents,
                max_steps=steps,
                seeds=list(range(runs)),
                output_dir=f"{output_dir}/exp2_comparison",
            )
            all_results, comparisons = asyncio.run(exp.run())

            console.print("\n[bold green]Significant Differences:[/bold green]")
            for comp in comparisons:
                if comp.significant:
                    console.print(
                        f"  {comp.metric_name}: {comp.value_a:.3f} vs {comp.value_b:.3f} "
                        f"(p={comp.p_value:.4f})"
                    )

        elif exp_type == "environmental":
            console.print(f"\n[bold]Running Scarcity vs Abundance for {model}[/bold]")
            exp = EnvironmentalComparison(
                model_name=model,
                n_agents=agents,
                max_steps=steps,
                seeds=list(range(runs)),
                output_dir=f"{output_dir}/exp3_environmental",
            )
            all_results, comparisons = asyncio.run(exp.run())

            console.print("\n[bold green]Environmental Effects:[/bold green]")
            for comp in comparisons:
                direction = "+" if comp.difference > 0 else ""
                console.print(
                    f"  {comp.metric_name}: scarcity={comp.value_a:.3f}, "
                    f"abundance={comp.value_b:.3f} ({direction}{comp.difference:.3f})"
                )

        else:
            console.print(f"[red]Unknown experiment type: {exp_type}[/red]")
            console.print("Available: baseline, model_comparison, environmental")
            sys.exit(1)

        console.print(f"\n[green]Results saved to {output_dir}[/green]")

    except Exception as e:
        console.print(f"[red]Experiment failed: {e}[/red]")
        if debug:
            console.print_exception()
        sys.exit(1)


@app.command()
def validate():
    """Validate installation and configuration"""
    console.print("🔍 [bold blue]Validating installation...[/bold blue]")

    try:
        # Test imports
        import mesa
        import torch
        import transformers

        console.print("✅ Core dependencies imported successfully")

        # Test mesa-frames
        try:
            import mesa_frames

            console.print("✅ Mesa-frames available")
        except ImportError:
            console.print("⚠️  Mesa-frames not available, will use fallback")

        # Test GPU availability
        if torch.cuda.is_available():
            console.print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            console.print("⚠️  CUDA not available, using CPU")

        console.print("🎉 [bold green]Installation validation complete![/bold green]")

    except Exception as e:
        console.print(f"❌ [bold red]Validation failed: {e}[/bold red]")


def main():
    """Main entry point"""
    app()


if __name__ == "__main__":
    main()
