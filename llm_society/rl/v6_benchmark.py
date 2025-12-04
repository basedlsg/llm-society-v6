"""
V6 Benchmark Runner

Implements the V6 experimental protocol:
- 10 episodes per policy with fixed random seeds
- 3 model families (Gemini, Groq, OpenAI)
- 7 scaffold conditions per model
- Enhanced metrics logging
- Stag Hunt cooperative gathering mechanics

Usage:
    python -m llm_society.rl.v6_benchmark --episodes 10 --model gemini
    python -m llm_society.rl.v6_benchmark --episodes 10 --all-models
"""

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

# IMPORTANT: Configure API keys BEFORE importing policy modules
_gemini_api_key = os.environ.get("GOOGLE_API_KEY")
if _gemini_api_key:
    try:
        import google.generativeai as genai
        genai.configure(api_key=_gemini_api_key)
    except ImportError:
        pass

from llm_society.rl.atropos_env import SurvivalWorld, SurvivalWorldConfig
from llm_society.rl.v6_prompts import V6State, ACTION_NAMES_V6, V6_SCAFFOLDS
from llm_society.rl.v6_policy import (
    V6GeminiPolicy,
    V6GroqPolicy,
    V6OpenAIPolicy,
    V6PolicyConfig,
    create_v6_policy,
    V6_MODEL_FAMILIES,
)
from llm_society.rl.trainer import SimplePolicy, PolicyConfig, obs_to_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# V6 WORLD CONFIGURATION
# =============================================================================

def create_v6_world_config(max_steps: int = 200) -> SurvivalWorldConfig:
    """Create V6 world configuration with Stag Hunt enabled."""
    return SurvivalWorldConfig(
        world_size=(20.0, 20.0),
        num_agents=2,
        max_steps=max_steps,
        energy_decay_per_step=0.02,
        rest_energy_gain=0.05,
        move_energy_cost=0.03,
        gather_energy_cost=0.04,
        initial_food=5,
        food_consumption_interval=10,
        starvation_health_penalty=0.1,
        social_radius=3.0,
        social_bonus=0.005,
        enable_social_bonus=True,
        # V6: Stag Hunt
        enable_stag_hunt=True,
        stag_hunt_radius=2.0,
        stag_hunt_food_bonus=5,
    )


# =============================================================================
# V6 FIXED SEEDS (10 seeds for statistical validity)
# =============================================================================

V6_SEEDS = [
    478163327,
    107420369,
    1181241943,
    1051802512,
    958682846,
    1298350006,
    843291752,
    2014786323,
    659142018,
    1742398556,
]


def generate_fixed_seeds(num_episodes: int, base_seed: int = 42) -> List[int]:
    """Generate fixed seeds for reproducibility across policies."""
    if num_episodes <= 10:
        return V6_SEEDS[:num_episodes]
    # Generate more if needed
    rng = random.Random(base_seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(num_episodes)]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def obs_to_v6_state(
    obs: Dict[str, Any],
    other_obs: Dict[str, Any],
    max_steps: int = 200
) -> V6State:
    """Convert observation dicts to V6State."""
    dx = obs["position_x"] - other_obs["position_x"]
    dy = obs["position_y"] - other_obs["position_y"]
    distance = (dx**2 + dy**2) ** 0.5

    return V6State(
        energy=obs["energy"],
        health=obs["health"],
        food=obs["food"],
        position_x=obs["position_x"],
        position_y=obs["position_y"],
        step=obs["step"],
        max_steps=max_steps,
        distance_to_other=distance,
        other_agent_nearby=distance <= 3.0,
        other_position_x=other_obs["position_x"],
        other_position_y=other_obs["position_y"],
    )


def action_dict_v6(action_idx: int, obs: Dict[str, Any], other_pos: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """Convert action index to action dict for V6."""
    if action_idx >= len(ACTION_NAMES_V6):
        action_idx = 0

    action_type = ACTION_NAMES_V6[action_idx]

    if action_type == "move_to":
        if other_pos:
            return {"type": "move_to", "params": {"x": other_pos[0], "y": other_pos[1]}}
        else:
            return {
                "type": "move_to",
                "params": {
                    "x": random.uniform(0, 20),
                    "y": random.uniform(0, 20),
                }
            }
    else:
        return {"type": action_type, "params": {}}


# =============================================================================
# V6 EPISODE RUNNER
# =============================================================================

def run_v6_episode(
    world: SurvivalWorld,
    policy_a,
    policy_b,
    seed: int,
    policy_a_name: str = "policy_a",
    policy_b_name: str = "policy_b",
) -> Dict[str, Any]:
    """
    Run a single V6 episode with two agents.

    Uses step_v5 for simultaneous action resolution (Stag Hunt).

    Returns:
        Episode results including per-step logs and enhanced metrics
    """
    # Reset world with seed
    world.reset(seed=seed)

    # Reset policies if they have reset method
    if hasattr(policy_a, 'reset'):
        policy_a.reset()
    if hasattr(policy_b, 'reset'):
        policy_b.reset()

    # Get initial observations
    obs_a = world._get_observation("agent_0")
    obs_b = world._get_observation("agent_1")

    # Tracking
    trajectory_a = []
    trajectory_b = []
    rewards_a = []
    rewards_b = []
    cooperative_gathers = 0
    last_reward_a = 0.0
    last_reward_b = 0.0
    last_energy_a = obs_a["energy"]
    last_energy_b = obs_b["energy"]
    last_food_a = obs_a["food"]
    last_food_b = obs_b["food"]

    # V6 Enhanced tracking
    energy_curve_a = [obs_a["energy"]]
    energy_curve_b = [obs_b["energy"]]
    food_curve_a = [obs_a["food"]]
    food_curve_b = [obs_b["food"]]

    done = False
    step = 0

    # Progress logging
    log_interval = 20  # Log every 20 steps

    while not done and step < world.config.max_steps:
        # Convert observations to V6State
        state_a = obs_to_v6_state(obs_a, obs_b, world.config.max_steps)
        state_b = obs_to_v6_state(obs_b, obs_a, world.config.max_steps)

        # Get actions from policies
        if hasattr(policy_a, 'select_action'):
            if hasattr(policy_a, 'config') and hasattr(policy_a.config, 'memory_enabled'):
                # V6 LLM policy
                action_idx_a, action_dict_a = policy_a.select_action(
                    state_a, last_reward_a, last_energy_a, last_food_a
                )
            else:
                # Legacy policy
                state_vec_a = obs_to_state(obs_a)
                action_idx_a, _ = policy_a.select_action(state_vec_a)
                other_pos = (obs_b["position_x"], obs_b["position_y"])
                action_dict_a = action_dict_v6(action_idx_a, obs_a, other_pos)
        else:
            # Heuristic or random policy (callable)
            action_dict_a = policy_a(obs_a)
            action_idx_a = ACTION_NAMES_V6.index(action_dict_a.get("type", "rest")) if action_dict_a.get("type") in ACTION_NAMES_V6 else 0

        if hasattr(policy_b, 'select_action'):
            if hasattr(policy_b, 'config') and hasattr(policy_b.config, 'memory_enabled'):
                action_idx_b, action_dict_b = policy_b.select_action(
                    state_b, last_reward_b, last_energy_b, last_food_b
                )
            else:
                state_vec_b = obs_to_state(obs_b)
                action_idx_b, _ = policy_b.select_action(state_vec_b)
                other_pos = (obs_a["position_x"], obs_a["position_y"])
                action_dict_b = action_dict_v6(action_idx_b, obs_b, other_pos)
        else:
            action_dict_b = policy_b(obs_b)
            action_idx_b = ACTION_NAMES_V6.index(action_dict_b.get("type", "rest")) if action_dict_b.get("type") in ACTION_NAMES_V6 else 0

        # Execute simultaneous step (V6 with Stag Hunt)
        results = world.step_v5({
            "agent_0": action_dict_a,
            "agent_1": action_dict_b,
        })

        obs_a, reward_a, done_a, info_a = results["agent_0"]
        obs_b, reward_b, done_b, info_b = results["agent_1"]

        # Track cooperative gathers
        if info_a.get("cooperative_gather", False):
            cooperative_gathers += 1

        # Log trajectory
        trajectory_a.append({
            "step": step,
            "state": asdict(state_a),
            "action": action_dict_a.get("type", "rest"),
            "action_params": action_dict_a.get("params", {}),
            "reward": reward_a,
            "cooperative_gather": info_a.get("cooperative_gather", False),
            "food_gained": info_a.get("food_gained", 0),
            "distance_to_other": state_a.distance_to_other,
            "energy_after": obs_a["energy"],
            "food_after": obs_a["food"],
        })

        trajectory_b.append({
            "step": step,
            "state": asdict(state_b),
            "action": action_dict_b.get("type", "rest"),
            "action_params": action_dict_b.get("params", {}),
            "reward": reward_b,
            "cooperative_gather": info_b.get("cooperative_gather", False),
            "food_gained": info_b.get("food_gained", 0),
            "distance_to_other": state_b.distance_to_other,
            "energy_after": obs_b["energy"],
            "food_after": obs_b["food"],
        })

        # Update curves
        energy_curve_a.append(obs_a["energy"])
        energy_curve_b.append(obs_b["energy"])
        food_curve_a.append(obs_a["food"])
        food_curve_b.append(obs_b["food"])

        rewards_a.append(reward_a)
        rewards_b.append(reward_b)
        last_reward_a = reward_a
        last_reward_b = reward_b
        last_energy_a = obs_a["energy"]
        last_energy_b = obs_b["energy"]
        last_food_a = obs_a["food"]
        last_food_b = obs_b["food"]

        done = done_a or done_b
        step += 1

        # Progress logging
        if step % log_interval == 0 or done:
            print(f"    Step {step}/{world.config.max_steps}: "
                  f"A({action_dict_a.get('type', '?')[:4]}) e={obs_a['energy']:.2f} f={obs_a['food']} | "
                  f"B({action_dict_b.get('type', '?')[:4]}) e={obs_b['energy']:.2f} f={obs_b['food']}", flush=True)

    # Determine death cause
    def get_death_cause(agent_id):
        agent = world.agents[agent_id]
        if agent.energy <= 0:
            return "energy"
        elif agent.health <= 0:
            return "health"
        return None

    # Compile results
    results = {
        "seed": seed,
        "policy_a": policy_a_name,
        "policy_b": policy_b_name,
        "episode_length": step,
        "agent_0": {
            "total_reward": sum(rewards_a),
            "survived": world.agents["agent_0"].health > 0 and world.agents["agent_0"].energy > 0,
            "death_cause": get_death_cause("agent_0"),
            "final_health": world.agents["agent_0"].health,
            "final_energy": world.agents["agent_0"].energy,
            "final_food": world.agents["agent_0"].food,
            "trajectory": trajectory_a,
            "action_counts": count_actions(trajectory_a),
            "energy_curve": energy_curve_a,
            "food_curve": food_curve_a,
        },
        "agent_1": {
            "total_reward": sum(rewards_b),
            "survived": world.agents["agent_1"].health > 0 and world.agents["agent_1"].energy > 0,
            "death_cause": get_death_cause("agent_1"),
            "final_health": world.agents["agent_1"].health,
            "final_energy": world.agents["agent_1"].energy,
            "final_food": world.agents["agent_1"].food,
            "trajectory": trajectory_b,
            "action_counts": count_actions(trajectory_b),
            "energy_curve": energy_curve_b,
            "food_curve": food_curve_b,
        },
        "cooperative_gathers": cooperative_gathers,
    }

    # Add raw LLM logs if available
    if hasattr(policy_a, 'get_step_logs'):
        results["agent_0"]["llm_logs"] = policy_a.get_step_logs()
        results["agent_0"]["llm_stats"] = policy_a.get_stats()
    if hasattr(policy_b, 'get_step_logs'):
        results["agent_1"]["llm_logs"] = policy_b.get_step_logs()
        results["agent_1"]["llm_stats"] = policy_b.get_stats()

    return results


def count_actions(trajectory: List[Dict]) -> Dict[str, int]:
    """Count action occurrences in trajectory."""
    counts = {}
    for step in trajectory:
        action = step.get("action", "rest")
        counts[action] = counts.get(action, 0) + 1
    return counts


# =============================================================================
# POLICY DEFINITIONS
# =============================================================================

def heuristic_policy_v6(obs: Dict[str, Any]) -> Dict[str, Any]:
    """V6 heuristic policy - balances rest and gather."""
    if obs["energy"] < 0.3:
        return {"type": "rest", "params": {}}
    if obs["food"] <= 2:
        return {"type": "gather_resources", "params": {}}
    if obs["energy"] < 0.5:
        return {"type": "rest", "params": {}}
    return {"type": "gather_resources", "params": {}}


def random_policy_v6(obs: Dict[str, Any]) -> Dict[str, Any]:
    """V6 random policy."""
    action_type = random.choice(ACTION_NAMES_V6)
    if action_type == "move_to":
        return {
            "type": "move_to",
            "params": {
                "x": random.uniform(0, 20),
                "y": random.uniform(0, 20),
            }
        }
    return {"type": action_type, "params": {}}


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_v6_benchmark(
    num_episodes: int = 10,
    model_family: Optional[str] = None,  # "gemini", "groq", "openai", or None for all
    scaffolds: Optional[List[str]] = None,  # List of scaffolds or None for all
    output_dir: str = "runs/survivalworld_v6",
    run_baselines: bool = True,
    run_rl: bool = True,
    max_steps: int = 200,  # Max steps per episode
) -> Dict[str, Any]:
    """
    Run V6 benchmark with specified configuration.

    Args:
        num_episodes: Number of episodes per condition (default 10)
        model_family: Which model to run (gemini/groq/openai) or None for all
        scaffolds: List of scaffolds to run or None for all 7
        output_dir: Directory to save results
        run_baselines: Whether to run heuristic/random baselines
        run_rl: Whether to run RL v3 baseline

    Returns:
        Summary of all benchmark results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate fixed seeds
    seeds = generate_fixed_seeds(num_episodes)

    # Create V6 world
    world_config = create_v6_world_config(max_steps=max_steps)
    world = SurvivalWorld(world_config)

    logger.info("=" * 70)
    logger.info("V6 BENCHMARK - SCAFFOLD EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Episodes per condition: {num_episodes}")
    logger.info(f"Seeds: {seeds[:5]}... (fixed for reproducibility)")
    logger.info(f"Model family: {model_family or 'ALL'}")
    logger.info(f"Stag Hunt: radius={world_config.stag_hunt_radius}, bonus=+{world_config.stag_hunt_food_bonus}")

    all_results = {}

    # ==========================================================================
    # BASELINE POLICIES (Heuristic, Random)
    # ==========================================================================

    if run_baselines:
        logger.info("\n" + "=" * 70)
        logger.info("BASELINE POLICIES")
        logger.info("=" * 70)

        # Heuristic vs Heuristic
        logger.info("\nRunning: Heuristic vs Heuristic")
        heur_results = []
        for i, seed in enumerate(seeds):
            result = run_v6_episode(
                world, heuristic_policy_v6, heuristic_policy_v6,
                seed, "heuristic", "heuristic"
            )
            heur_results.append(result)
            if (i + 1) % 5 == 0:
                logger.info(f"  Episode {i + 1}/{num_episodes}")

        all_results["heuristic_vs_heuristic"] = heur_results
        save_results(output_path, "heuristic_vs_heuristic", heur_results, world_config)

        # Random vs Random
        logger.info("\nRunning: Random vs Random")
        random_results = []
        for i, seed in enumerate(seeds):
            result = run_v6_episode(
                world, random_policy_v6, random_policy_v6,
                seed, "random", "random"
            )
            random_results.append(result)
            if (i + 1) % 5 == 0:
                logger.info(f"  Episode {i + 1}/{num_episodes}")

        all_results["random_vs_random"] = random_results
        save_results(output_path, "random_vs_random", random_results, world_config)

    # ==========================================================================
    # RL POLICY
    # ==========================================================================

    rl_policy = None
    if run_rl:
        logger.info("\n" + "=" * 70)
        logger.info("RL v3 POLICY")
        logger.info("=" * 70)

        rl_policy_path = Path("./training_results_v3/policy_v3_final.npz")
        if rl_policy_path.exists():
            rl_config = PolicyConfig(state_dim=8, action_dim=4, hidden_dim=32)
            rl_policy = SimplePolicy(rl_config)
            rl_policy.load(str(rl_policy_path))
            logger.info(f"Loaded RL v3 policy from {rl_policy_path}")

            # RL vs RL
            logger.info("\nRunning: RL v3 vs RL v3")
            rl_results = []
            for i, seed in enumerate(seeds):
                result = run_v6_episode(
                    world, rl_policy, rl_policy,
                    seed, "rl_v3", "rl_v3"
                )
                rl_results.append(result)
                if (i + 1) % 5 == 0:
                    logger.info(f"  Episode {i + 1}/{num_episodes}")

            all_results["rl_v3_vs_rl_v3"] = rl_results
            save_results(output_path, "rl_v3_vs_rl_v3", rl_results, world_config)
        else:
            logger.warning(f"RL v3 policy not found at {rl_policy_path}")

    # ==========================================================================
    # LLM POLICIES
    # ==========================================================================

    # Determine which models to run
    models_to_run = []
    if model_family:
        models_to_run = [model_family]
    else:
        # Check which API keys are available
        if os.environ.get("GOOGLE_API_KEY"):
            models_to_run.append("gemini")
        if os.environ.get("GROQ_API_KEY"):
            models_to_run.append("groq")
        if os.environ.get("OPENAI_API_KEY"):
            models_to_run.append("openai")

    # Determine which scaffolds to run
    scaffolds_to_run = scaffolds or V6_SCAFFOLDS

    for model in models_to_run:
        logger.info("\n" + "=" * 70)
        logger.info(f"{model.upper()} LLM POLICIES")
        logger.info("=" * 70)

        for scaffold in scaffolds_to_run:
            condition_name = f"{model}_{scaffold}"
            logger.info(f"\nRunning: {condition_name} vs {condition_name}")

            try:
                policy = create_v6_policy(backend=model, scaffold=scaffold)

                results = []
                for i, seed in enumerate(seeds):
                    result = run_v6_episode(
                        world, policy, policy,
                        seed, condition_name, condition_name
                    )
                    results.append(result)
                    if (i + 1) % 5 == 0:
                        logger.info(f"  Episode {i + 1}/{num_episodes}")

                all_results[f"{condition_name}_vs_{condition_name}"] = results
                save_results(output_path, f"{condition_name}_vs_{condition_name}", results, world_config)

                # Print policy stats
                stats = policy.get_stats()
                logger.info(f"  Stats: {stats}")

            except Exception as e:
                logger.error(f"Error running {condition_name}: {e}")
                import traceback
                traceback.print_exc()

        # Mixed: RL vs LLM (baseline scaffold)
        if rl_policy is not None and "baseline_nomem" in scaffolds_to_run:
            logger.info(f"\nRunning: RL v3 vs {model} baseline_nomem")
            try:
                llm_policy = create_v6_policy(backend=model, scaffold="baseline_nomem")

                results = []
                for i, seed in enumerate(seeds):
                    result = run_v6_episode(
                        world, rl_policy, llm_policy,
                        seed, "rl_v3", f"{model}_baseline_nomem"
                    )
                    results.append(result)
                    if (i + 1) % 5 == 0:
                        logger.info(f"  Episode {i + 1}/{num_episodes}")

                all_results[f"rl_v3_vs_{model}_baseline"] = results
                save_results(output_path, f"rl_v3_vs_{model}_baseline", results, world_config)

            except Exception as e:
                logger.error(f"Error running RL vs {model}: {e}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("V6 BENCHMARK COMPLETE")
    logger.info("=" * 70)

    print_summary(all_results)

    return all_results


def save_results(
    output_path: Path,
    name: str,
    results: List[Dict],
    world_config: SurvivalWorldConfig
):
    """Save benchmark results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"{timestamp}_v6_{name}.json"

    # Compute summary statistics
    summary = compute_summary(results)

    output_data = {
        "version": "v6",
        "name": name,
        "timestamp": datetime.now().isoformat(),
        "world_config": {
            "world_size": list(world_config.world_size),
            "num_agents": world_config.num_agents,
            "max_steps": world_config.max_steps,
            "enable_stag_hunt": world_config.enable_stag_hunt,
            "stag_hunt_radius": world_config.stag_hunt_radius,
            "stag_hunt_food_bonus": world_config.stag_hunt_food_bonus,
            "social_bonus": world_config.social_bonus,
        },
        "num_episodes": len(results),
        "summary": summary,
        "episodes": results,
    }

    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info(f"Saved results to: {filename}")


def compute_summary(results: List[Dict]) -> Dict[str, Any]:
    """Compute summary statistics from episode results."""
    if not results:
        return {}

    a0_rewards = [r["agent_0"]["total_reward"] for r in results]
    a1_rewards = [r["agent_1"]["total_reward"] for r in results]
    a0_survived = [r["agent_0"]["survived"] for r in results]
    a1_survived = [r["agent_1"]["survived"] for r in results]
    coop_gathers = [r["cooperative_gathers"] for r in results]
    ep_lengths = [r["episode_length"] for r in results]

    # Death cause analysis
    a0_deaths_energy = sum(1 for r in results if r["agent_0"]["death_cause"] == "energy")
    a0_deaths_health = sum(1 for r in results if r["agent_0"]["death_cause"] == "health")
    a1_deaths_energy = sum(1 for r in results if r["agent_1"]["death_cause"] == "energy")
    a1_deaths_health = sum(1 for r in results if r["agent_1"]["death_cause"] == "health")

    # Action distribution
    a0_actions = {}
    a1_actions = {}
    for r in results:
        for action, count in r["agent_0"]["action_counts"].items():
            a0_actions[action] = a0_actions.get(action, 0) + count
        for action, count in r["agent_1"]["action_counts"].items():
            a1_actions[action] = a1_actions.get(action, 0) + count

    # Token stats (if available)
    total_prompt_tokens = 0
    total_response_tokens = 0
    for r in results:
        if "llm_stats" in r["agent_0"]:
            total_prompt_tokens += r["agent_0"]["llm_stats"].get("total_prompt_tokens", 0)
            total_response_tokens += r["agent_0"]["llm_stats"].get("total_response_tokens", 0)
        if "llm_stats" in r["agent_1"]:
            total_prompt_tokens += r["agent_1"]["llm_stats"].get("total_prompt_tokens", 0)
            total_response_tokens += r["agent_1"]["llm_stats"].get("total_response_tokens", 0)

    return {
        "agent_0": {
            "mean_reward": float(np.mean(a0_rewards)),
            "std_reward": float(np.std(a0_rewards)),
            "survival_rate": float(np.mean(a0_survived)),
            "deaths_by_energy": a0_deaths_energy,
            "deaths_by_health": a0_deaths_health,
            "action_distribution": a0_actions,
        },
        "agent_1": {
            "mean_reward": float(np.mean(a1_rewards)),
            "std_reward": float(np.std(a1_rewards)),
            "survival_rate": float(np.mean(a1_survived)),
            "deaths_by_energy": a1_deaths_energy,
            "deaths_by_health": a1_deaths_health,
            "action_distribution": a1_actions,
        },
        "episode": {
            "mean_length": float(np.mean(ep_lengths)),
            "std_length": float(np.std(ep_lengths)),
            "min_length": int(np.min(ep_lengths)),
            "max_length": int(np.max(ep_lengths)),
        },
        "cooperative_gathers": {
            "mean": float(np.mean(coop_gathers)),
            "total": sum(coop_gathers),
            "episodes_with_coop": sum(1 for c in coop_gathers if c > 0),
        },
        "tokens": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_response_tokens": total_response_tokens,
        },
    }


def print_summary(all_results: Dict[str, List[Dict]]):
    """Print summary of all benchmark results."""
    print("\n" + "=" * 90)
    print("V6 BENCHMARK SUMMARY")
    print("=" * 90)

    for name, results in all_results.items():
        if not results:
            continue

        summary = compute_summary(results)
        a0 = summary["agent_0"]
        a1 = summary["agent_1"]
        ep = summary["episode"]
        coop = summary["cooperative_gathers"]

        print(f"\n{name}:")
        print(f"  Episodes: {len(results)}, Length: {ep['mean_length']:.1f} (min={ep['min_length']}, max={ep['max_length']})")
        print(f"  Agent 0: reward={a0['mean_reward']:.3f}±{a0['std_reward']:.3f}, survival={a0['survival_rate']*100:.1f}%")
        print(f"           deaths: energy={a0['deaths_by_energy']}, health={a0['deaths_by_health']}")
        print(f"  Agent 1: reward={a1['mean_reward']:.3f}±{a1['std_reward']:.3f}, survival={a1['survival_rate']*100:.1f}%")
        print(f"           deaths: energy={a1['deaths_by_energy']}, health={a1['deaths_by_health']}")
        print(f"  Cooperative gathers: {coop['mean']:.1f} avg, {coop['episodes_with_coop']}/{len(results)} episodes")
        print(f"  Actions (a0): {a0['action_distribution']}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="V6 Benchmark Runner")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per condition")
    parser.add_argument("--model", type=str, choices=["gemini", "groq", "openai"],
                        help="Run specific model (default: all available)")
    parser.add_argument("--scaffold", type=str, nargs="+",
                        help="Run specific scaffolds (default: all)")
    parser.add_argument("--output", type=str, default="runs/survivalworld_v6",
                        help="Output directory")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip heuristic/random baselines")
    parser.add_argument("--no-rl", action="store_true",
                        help="Skip RL baseline")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max steps per episode (default: 200)")

    args = parser.parse_args()

    run_v6_benchmark(
        num_episodes=args.episodes,
        model_family=args.model,
        scaffolds=args.scaffold,
        output_dir=args.output,
        run_baselines=not args.no_baselines,
        run_rl=not args.no_rl,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
