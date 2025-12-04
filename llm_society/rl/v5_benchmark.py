"""
V5 Benchmark Runner

Implements the V5 experimental protocol:
- 50 episodes per policy with fixed random seeds
- Identical initial conditions across all policies
- Full logging of raw LLM outputs
- Stag Hunt cooperative gathering mechanics

Usage:
    python -m llm_society.rl.v5_benchmark --episodes 50 --gemini
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

# IMPORTANT: Configure Gemini API key BEFORE importing policy modules
# This ensures the genai library is configured with the correct key
# before any other modules try to use it
_gemini_api_key = os.environ.get("GOOGLE_API_KEY")
if _gemini_api_key:
    try:
        import google.generativeai as genai
        genai.configure(api_key=_gemini_api_key)
    except ImportError:
        pass  # genai not installed

from llm_society.rl.atropos_env import SurvivalWorld, SurvivalWorldConfig
from llm_society.rl.v5_prompts import V5State, ACTION_NAMES_V5
from llm_society.rl.v5_policy import (
    V5GeminiPolicy,
    V5GroqPolicy,
    V5PolicyConfig,
    create_v5_policy,
)
from llm_society.rl.trainer import SimplePolicy, PolicyConfig, obs_to_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# V5 WORLD CONFIGURATION
# =============================================================================

def create_v5_world_config() -> SurvivalWorldConfig:
    """Create V5 world configuration with Stag Hunt enabled."""
    return SurvivalWorldConfig(
        world_size=(20.0, 20.0),
        num_agents=2,
        max_steps=200,
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
        # V5: Stag Hunt
        enable_stag_hunt=True,
        stag_hunt_radius=2.0,
        stag_hunt_food_bonus=5,
    )


# =============================================================================
# FIXED SEEDS FOR REPRODUCIBILITY
# =============================================================================

def generate_fixed_seeds(num_episodes: int, base_seed: int = 42) -> List[int]:
    """Generate fixed seeds for reproducibility across policies."""
    rng = random.Random(base_seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(num_episodes)]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def obs_to_v5_state(
    obs: Dict[str, Any],
    other_obs: Dict[str, Any],
    max_steps: int = 200
) -> V5State:
    """Convert observation dicts to V5State."""
    # Calculate distance to other agent
    dx = obs["position_x"] - other_obs["position_x"]
    dy = obs["position_y"] - other_obs["position_y"]
    distance = (dx**2 + dy**2) ** 0.5

    return V5State(
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


def action_dict_v5(action_idx: int, obs: Dict[str, Any], other_pos: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """Convert action index to action dict for V5."""
    if action_idx >= len(ACTION_NAMES_V5):
        action_idx = 0  # Default to rest

    action_type = ACTION_NAMES_V5[action_idx]

    if action_type == "move_to":
        if other_pos:
            # Move toward other agent
            return {"type": "move_to", "params": {"x": other_pos[0], "y": other_pos[1]}}
        else:
            # Random direction
            return {
                "type": "move_to",
                "params": {
                    "x": random.uniform(0, obs.get("world_width", 20)),
                    "y": random.uniform(0, obs.get("world_height", 20)),
                }
            }
    else:
        return {"type": action_type, "params": {}}


# =============================================================================
# V5 EPISODE RUNNER
# =============================================================================

def run_v5_episode(
    world: SurvivalWorld,
    policy_a,
    policy_b,
    seed: int,
    policy_a_name: str = "policy_a",
    policy_b_name: str = "policy_b",
) -> Dict[str, Any]:
    """
    Run a single V5 episode with two agents.

    Uses step_v5 for simultaneous action resolution (Stag Hunt).

    Returns:
        Episode results including per-step logs
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

    done = False
    step = 0

    while not done and step < world.config.max_steps:
        # Convert observations to V5State
        state_a = obs_to_v5_state(obs_a, obs_b, world.config.max_steps)
        state_b = obs_to_v5_state(obs_b, obs_a, world.config.max_steps)

        # Get actions from policies
        if hasattr(policy_a, 'select_action'):
            # LLM policy
            if hasattr(policy_a, 'config') and hasattr(policy_a.config, 'memory_enabled'):
                # V5 LLM policy
                action_idx_a, action_dict_a = policy_a.select_action(state_a, last_reward_a)
            else:
                # Legacy policy
                state_vec_a = obs_to_state(obs_a)
                action_idx_a, _ = policy_a.select_action(state_vec_a)
                other_pos = (obs_b["position_x"], obs_b["position_y"])
                action_dict_a = action_dict_v5(action_idx_a, obs_a, other_pos)
        else:
            # Heuristic or random policy (callable)
            action_dict_a = policy_a(obs_a)
            action_idx_a = ACTION_NAMES_V5.index(action_dict_a.get("type", "rest")) if action_dict_a.get("type") in ACTION_NAMES_V5 else 0

        if hasattr(policy_b, 'select_action'):
            if hasattr(policy_b, 'config') and hasattr(policy_b.config, 'memory_enabled'):
                action_idx_b, action_dict_b = policy_b.select_action(state_b, last_reward_b)
            else:
                state_vec_b = obs_to_state(obs_b)
                action_idx_b, _ = policy_b.select_action(state_vec_b)
                other_pos = (obs_a["position_x"], obs_a["position_y"])
                action_dict_b = action_dict_v5(action_idx_b, obs_b, other_pos)
        else:
            action_dict_b = policy_b(obs_b)
            action_idx_b = ACTION_NAMES_V5.index(action_dict_b.get("type", "rest")) if action_dict_b.get("type") in ACTION_NAMES_V5 else 0

        # Execute simultaneous step (V5 with Stag Hunt)
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
        })

        rewards_a.append(reward_a)
        rewards_b.append(reward_b)
        last_reward_a = reward_a
        last_reward_b = reward_b

        done = done_a or done_b
        step += 1

    # Compile results
    results = {
        "seed": seed,
        "policy_a": policy_a_name,
        "policy_b": policy_b_name,
        "episode_length": step,
        "agent_0": {
            "total_reward": sum(rewards_a),
            "survived": world.agents["agent_0"].health > 0 and world.agents["agent_0"].energy > 0,
            "final_health": world.agents["agent_0"].health,
            "final_energy": world.agents["agent_0"].energy,
            "final_food": world.agents["agent_0"].food,
            "trajectory": trajectory_a,
            "action_counts": count_actions(trajectory_a),
        },
        "agent_1": {
            "total_reward": sum(rewards_b),
            "survived": world.agents["agent_1"].health > 0 and world.agents["agent_1"].energy > 0,
            "final_health": world.agents["agent_1"].health,
            "final_energy": world.agents["agent_1"].energy,
            "final_food": world.agents["agent_1"].food,
            "trajectory": trajectory_b,
            "action_counts": count_actions(trajectory_b),
        },
        "cooperative_gathers": cooperative_gathers,
    }

    # Add raw LLM logs if available
    if hasattr(policy_a, 'get_step_logs'):
        results["agent_0"]["llm_logs"] = policy_a.get_step_logs()
    if hasattr(policy_b, 'get_step_logs'):
        results["agent_1"]["llm_logs"] = policy_b.get_step_logs()

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

def heuristic_policy_v5(obs: Dict[str, Any]) -> Dict[str, Any]:
    """V5 heuristic policy (same as v4)."""
    if obs["energy"] < 0.3:
        return {"type": "rest", "params": {}}
    if obs["food"] <= 2:
        return {"type": "gather_resources", "params": {}}
    return {"type": "rest", "params": {}}


def random_policy_v5(obs: Dict[str, Any]) -> Dict[str, Any]:
    """V5 random policy."""
    action_type = random.choice(ACTION_NAMES_V5)
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

def run_v5_benchmark(
    num_episodes: int = 50,
    use_gemini: bool = False,
    use_groq: bool = False,
    output_dir: str = "runs/survivalworld_v5",
) -> Dict[str, Any]:
    """
    Run V5 benchmark with all policy configurations.

    Args:
        num_episodes: Number of episodes per policy configuration
        use_gemini: Whether to run Gemini LLM policies
        use_groq: Whether to run Groq LLM policies
        output_dir: Directory to save results

    Returns:
        Summary of all benchmark results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate fixed seeds
    seeds = generate_fixed_seeds(num_episodes)

    # Create V5 world
    world_config = create_v5_world_config()
    world = SurvivalWorld(world_config)

    logger.info("=" * 70)
    logger.info("V5 BENCHMARK - STAG HUNT COOPERATIVE GATHERING")
    logger.info("=" * 70)
    logger.info(f"Episodes per policy: {num_episodes}")
    logger.info(f"Seeds: {seeds[:5]}... (fixed for reproducibility)")
    logger.info(f"Stag Hunt enabled: radius={world_config.stag_hunt_radius}, bonus=+{world_config.stag_hunt_food_bonus}")

    all_results = {}

    # ==========================================================================
    # BASELINE POLICIES (Heuristic, Random)
    # ==========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("BASELINE POLICIES")
    logger.info("=" * 70)

    # Heuristic vs Heuristic
    logger.info("\nRunning: Heuristic vs Heuristic")
    heur_results = []
    for i, seed in enumerate(seeds):
        result = run_v5_episode(
            world, heuristic_policy_v5, heuristic_policy_v5,
            seed, "heuristic", "heuristic"
        )
        heur_results.append(result)
        if (i + 1) % 10 == 0:
            logger.info(f"  Episode {i + 1}/{num_episodes}")

    all_results["heuristic_vs_heuristic"] = heur_results
    save_results(output_path, "heuristic_vs_heuristic", heur_results, world_config)

    # Random vs Random
    logger.info("\nRunning: Random vs Random")
    random_results = []
    for i, seed in enumerate(seeds):
        result = run_v5_episode(
            world, random_policy_v5, random_policy_v5,
            seed, "random", "random"
        )
        random_results.append(result)
        if (i + 1) % 10 == 0:
            logger.info(f"  Episode {i + 1}/{num_episodes}")

    all_results["random_vs_random"] = random_results
    save_results(output_path, "random_vs_random", random_results, world_config)

    # ==========================================================================
    # RL POLICY
    # ==========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("RL v3 POLICY")
    logger.info("=" * 70)

    # Load RL v3 policy if available
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
            result = run_v5_episode(
                world, rl_policy, rl_policy,
                seed, "rl_v3", "rl_v3"
            )
            rl_results.append(result)
            if (i + 1) % 10 == 0:
                logger.info(f"  Episode {i + 1}/{num_episodes}")

        all_results["rl_v3_vs_rl_v3"] = rl_results
        save_results(output_path, "rl_v3_vs_rl_v3", rl_results, world_config)
    else:
        logger.warning(f"RL v3 policy not found at {rl_policy_path}")
        rl_policy = None

    # ==========================================================================
    # LLM POLICIES (Gemini)
    # ==========================================================================

    if use_gemini:
        logger.info("\n" + "=" * 70)
        logger.info("GEMINI LLM POLICIES")
        logger.info("=" * 70)

        prompt_formats = ["baseline", "explicit", "reasoning"]
        memory_conditions = [False, True]

        for prompt_format in prompt_formats:
            for memory_enabled in memory_conditions:
                condition_name = f"gemini_{prompt_format}_{'memory' if memory_enabled else 'nomem'}"
                logger.info(f"\nRunning: {condition_name} vs {condition_name}")

                try:
                    policy = create_v5_policy(
                        backend="gemini",
                        prompt_format=prompt_format,
                        memory_enabled=memory_enabled,
                    )

                    results = []
                    for i, seed in enumerate(seeds):
                        result = run_v5_episode(
                            world, policy, policy,
                            seed, condition_name, condition_name
                        )
                        results.append(result)
                        if (i + 1) % 10 == 0:
                            logger.info(f"  Episode {i + 1}/{num_episodes}")

                    all_results[f"{condition_name}_vs_{condition_name}"] = results
                    save_results(output_path, f"{condition_name}_vs_{condition_name}", results, world_config)

                    # Print policy stats
                    stats = policy.get_stats()
                    logger.info(f"  Stats: {stats}")

                except Exception as e:
                    logger.error(f"Error running {condition_name}: {e}")

        # Mixed: RL vs Gemini (baseline, no memory)
        if rl_policy is not None:
            logger.info("\nRunning: RL v3 vs Gemini (baseline, no memory)")
            try:
                gemini_policy = create_v5_policy(
                    backend="gemini",
                    prompt_format="baseline",
                    memory_enabled=False,
                )

                results = []
                for i, seed in enumerate(seeds):
                    result = run_v5_episode(
                        world, rl_policy, gemini_policy,
                        seed, "rl_v3", "gemini_baseline_nomem"
                    )
                    results.append(result)
                    if (i + 1) % 10 == 0:
                        logger.info(f"  Episode {i + 1}/{num_episodes}")

                all_results["rl_v3_vs_gemini_baseline"] = results
                save_results(output_path, "rl_v3_vs_gemini_baseline", results, world_config)

            except Exception as e:
                logger.error(f"Error running RL vs Gemini: {e}")

    # ==========================================================================
    # LLM POLICIES (Groq)
    # ==========================================================================

    if use_groq:
        logger.info("\n" + "=" * 70)
        logger.info("GROQ LLM POLICIES")
        logger.info("=" * 70)

        # Only run baseline prompt for Groq due to rate limits
        for memory_enabled in [False, True]:
            condition_name = f"groq_baseline_{'memory' if memory_enabled else 'nomem'}"
            logger.info(f"\nRunning: {condition_name} vs {condition_name}")

            try:
                policy = create_v5_policy(
                    backend="groq",
                    prompt_format="baseline",
                    memory_enabled=memory_enabled,
                )

                results = []
                for i, seed in enumerate(seeds):
                    result = run_v5_episode(
                        world, policy, policy,
                        seed, condition_name, condition_name
                    )
                    results.append(result)
                    if (i + 1) % 10 == 0:
                        logger.info(f"  Episode {i + 1}/{num_episodes}")

                all_results[f"{condition_name}_vs_{condition_name}"] = results
                save_results(output_path, f"{condition_name}_vs_{condition_name}", results, world_config)

            except Exception as e:
                logger.error(f"Error running {condition_name}: {e}")
                break  # Likely rate limited

    # ==========================================================================
    # SUMMARY
    # ==========================================================================

    logger.info("\n" + "=" * 70)
    logger.info("V5 BENCHMARK COMPLETE")
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
    filename = output_path / f"{timestamp}_v5_{name}.json"

    # Compute summary statistics
    summary = compute_summary(results)

    output_data = {
        "version": "v5",
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

    # Action distribution
    a0_actions = {}
    a1_actions = {}
    for r in results:
        for action, count in r["agent_0"]["action_counts"].items():
            a0_actions[action] = a0_actions.get(action, 0) + count
        for action, count in r["agent_1"]["action_counts"].items():
            a1_actions[action] = a1_actions.get(action, 0) + count

    return {
        "agent_0": {
            "mean_reward": np.mean(a0_rewards),
            "std_reward": np.std(a0_rewards),
            "survival_rate": np.mean(a0_survived),
            "action_distribution": a0_actions,
        },
        "agent_1": {
            "mean_reward": np.mean(a1_rewards),
            "std_reward": np.std(a1_rewards),
            "survival_rate": np.mean(a1_survived),
            "action_distribution": a1_actions,
        },
        "cooperative_gathers": {
            "mean": np.mean(coop_gathers),
            "total": sum(coop_gathers),
            "episodes_with_coop": sum(1 for c in coop_gathers if c > 0),
        },
    }


def print_summary(all_results: Dict[str, List[Dict]]):
    """Print summary of all benchmark results."""
    print("\n" + "=" * 80)
    print("V5 BENCHMARK SUMMARY")
    print("=" * 80)

    for name, results in all_results.items():
        if not results:
            continue

        summary = compute_summary(results)
        a0 = summary["agent_0"]
        a1 = summary["agent_1"]
        coop = summary["cooperative_gathers"]

        print(f"\n{name}:")
        print(f"  Agent 0: reward={a0['mean_reward']:.3f}±{a0['std_reward']:.3f}, survival={a0['survival_rate']*100:.1f}%")
        print(f"  Agent 1: reward={a1['mean_reward']:.3f}±{a1['std_reward']:.3f}, survival={a1['survival_rate']*100:.1f}%")
        print(f"  Cooperative gathers: {coop['mean']:.1f} avg, {coop['episodes_with_coop']}/{len(results)} episodes")
        print(f"  Action dist (a0): {a0['action_distribution']}")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="V5 Benchmark Runner")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per policy")
    parser.add_argument("--gemini", action="store_true", help="Run Gemini LLM policies")
    parser.add_argument("--groq", action="store_true", help="Run Groq LLM policies")
    parser.add_argument("--output", type=str, default="runs/survivalworld_v5", help="Output directory")

    args = parser.parse_args()

    run_v5_benchmark(
        num_episodes=args.episodes,
        use_gemini=args.gemini,
        use_groq=args.groq,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
