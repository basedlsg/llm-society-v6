from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Default values, can be overridden by a config file or CLI arguments


@dataclass
class LLMConfig:
    model_name: str = "gemini-pro"
    secondary_model_name: Optional[str] = None
    max_tokens: int = 150
    temperature: float = 0.7
    api_max_retries: int = 3
    api_base_backoff: float = 1.0
    api_max_backoff: float = 16.0
    cache_responses: bool = True
    rate_limit_per_second: int = 10
    max_cache_size: int = 5000  # Increased for better cache hit rate at scale
    request_timeout: float = 60.0
    batch_size: int = 10  # Number of requests to batch together
    batch_timeout: float = 0.1  # Max seconds to wait for batch to fill
    enable_request_deduplication: bool = True  # Deduplicate identical pending requests


@dataclass
class AgentConfig:
    count: int = 50
    movement_speed: float = 1.0
    social_radius: float = 3.0  # v1.1: Reduced from 10.0 to require proximity for talk_to
    memory_size: int = 20
    interaction_radius: float = 5.0
    initial_min_age: float = 20.0
    initial_max_age: float = 50.0
    initial_health: float = 1.0
    initial_employed_prob: float = 0.5
    initial_food: int = 5  # v1.1: Reduced from 10 to create survival pressure
    initial_currency: int = 500
    # Potentially: persona_options, etc.


@dataclass
class SimulationConfig:
    max_steps: int = 1000
    # World size reduced to 20x20 to ensure agents can interact
    # With social_radius=10, agents in 20x20 world will meet each other
    world_size: Tuple[int, int] = (20, 20)
    tick_rate: float = 0.1  # Seconds per simulation step, if using timed delays
    seed: Optional[int] = None  # For reproducibility
    autosave_enabled: bool = True
    autosave_interval_steps: Optional[int] = (
        100  # Save every 100 steps, None to disable interval
    )
    autosave_file_pattern: str = "autosave_step_{step}.json"
    autosave_directory: str = (
        "autosaves"  # Subdirectory within the main output.directory
    )
    # v1.1 Energy dynamics - increased decay, actions have costs
    energy_decay_per_step: float = 0.02  # v1.1: Increased from 0.005 to create real pressure
    rest_energy_gain: float = 0.05  # v1.1: Reduced from 0.2 (net +0.03/step if resting)
    move_energy_cost: float = 0.03  # v1.1: Movement costs energy
    talk_energy_cost: float = 0.02  # v1.1: Talking costs energy
    gather_energy_cost: float = 0.04  # v1.1: Gathering costs energy
    # v1.1 Food dynamics - faster decay, starvation penalty
    food_consumption_interval: int = 10  # v1.1: Reduced from 20 to create urgency
    starvation_health_penalty: float = 0.1  # v1.1: Health loss when food = 0


@dataclass
class OutputConfig:
    directory: str = "./results"
    metrics_file: str = "metrics.db"
    log_file: str = "simulation.log"
    generated_assets_dir: str = "generated_assets"  # For Point-E outputs
    database_url: Optional[str] = (
        "sqlite:///./llm_society_dynamic_data.db"  # For dynamic data like memories, transactions
    )
    # Database connection pool settings
    db_pool_size: int = 5  # Number of connections to keep in the pool
    db_max_overflow: int = 10  # Additional connections beyond pool_size
    db_pool_timeout: float = 30.0  # Seconds to wait for a connection


@dataclass
class PerformanceConfig:
    """Performance tuning configuration"""
    enable_gpu_acceleration: bool = False  # Enable FLAME GPU acceleration
    gpu_device_id: int = 0  # GPU device to use
    agent_batch_size: int = 100  # Agents to process per batch
    parallel_llm_requests: int = 5  # Max concurrent LLM requests
    memory_limit_mb: int = 4096  # Memory limit for agent data
    enable_profiling: bool = False  # Enable performance profiling
    profile_interval: int = 100  # Steps between profile snapshots
    async_database_writes: bool = True  # Use async database writes
    cache_agent_decisions: bool = True  # Cache recent agent decisions
    decision_cache_ttl: int = 10  # Steps before cached decisions expire


@dataclass
class MonitoringConfig:
    enable_metrics: bool = True
    metrics_interval: int = 10  # Steps per metrics collection
    # Add wandb or other monitoring tool configs if used


@dataclass
class AssetsConfig:  # New section for asset related configurations
    enable_generation: bool = (
        True  # To globally enable/disable asset generation by agents
    )
    # Could add more specific Point-E configs here if needed, e.g., model choice, quality etc.


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    assets: AssetsConfig = field(default_factory=AssetsConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    def to_dict(self) -> Dict[str, Any]:  # For saving config snapshot
        return asdict(self)

    @staticmethod
    def default() -> "Config":
        return Config()

    @staticmethod
    def load(config_path: str) -> "Config":
        try:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)

            # A more robust loading would involve mapping dict keys to dataclass fields,
            # possibly using a library like dacite or manually traversing.
            # For simplicity, this example assumes top-level keys match dataclass field names.

            # Example of manual nested dataclass creation:
            llm_conf = LLMConfig(**config_dict.get("llm", {}))
            agent_conf = AgentConfig(**config_dict.get("agents", {}))
            sim_conf_data = config_dict.get("simulation", {})
            # Ensure autosave_interval_steps is None if not present or explicitly null in YAML
            if (
                "autosave_interval_steps" in sim_conf_data
                and sim_conf_data["autosave_interval_steps"] is None
            ):
                pass  # It's already None or will be handled by dataclass default if key is missing
            elif "autosave_interval_steps" not in sim_conf_data:
                sim_conf_data["autosave_interval_steps"] = (
                    SimulationConfig.autosave_interval_steps
                )  # Use dataclass default

            sim_conf = SimulationConfig(**sim_conf_data)
            out_conf = OutputConfig(**config_dict.get("output", {}))
            mon_conf = MonitoringConfig(**config_dict.get("monitoring", {}))
            asset_conf = AssetsConfig(**config_dict.get("assets", {}))
            perf_conf = PerformanceConfig(**config_dict.get("performance", {}))

            return Config(
                llm=llm_conf,
                agents=agent_conf,
                simulation=sim_conf,
                output=out_conf,
                monitoring=mon_conf,
                assets=asset_conf,
                performance=perf_conf,
            )
        except FileNotFoundError:
            print(
                f"Warning: Config file {config_path} not found. Using default config."
            )
            return Config.default()
        except Exception as e:
            print(
                f"Error loading config file {config_path}: {e}. Using default config."
            )
            return Config.default()


if __name__ == "__main__":
    # Test default config
    default_config = Config.default()
    print("Default Config:")
    print(f"  LLM Model: {default_config.llm.model_name}")
    print(f"  Agent Count: {default_config.agents.count}")
    print(f"  Max Steps: {default_config.simulation.max_steps}")
    print(f"  Output Dir: {default_config.output.directory}")
    print(f"  Generated Assets Dir: {default_config.output.generated_assets_dir}")
    print(f"  Asset Generation Enabled: {default_config.assets.enable_generation}")

    # Test loading from a dummy YAML (if one were to exist)
    # Create a dummy yaml for testing
    dummy_yaml_content = """
llm:
  model_name: 'test_model'
agents:
  count: 10
output:
  directory: '/tmp/test_output'
  generated_assets_dir: '/tmp/test_output/assets'
assets:
  enable_generation: false
"""
    dummy_yaml_path = "dummy_config.yaml"
    with open(dummy_yaml_path, "w") as f:
        f.write(dummy_yaml_content)

    loaded_config = Config.load(dummy_yaml_path)
    print("\nLoaded Config (from dummy_config.yaml):")
    print(f"  LLM Model: {loaded_config.llm.model_name}")
    print(f"  Agent Count: {loaded_config.agents.count}")
    print(f"  Output Dir: {loaded_config.output.directory}")
    print(f"  Generated Assets Dir: {loaded_config.output.generated_assets_dir}")
    print(f"  Asset Generation Enabled: {loaded_config.assets.enable_generation}")

    import os

    os.remove(dummy_yaml_path)  # Clean up dummy file
