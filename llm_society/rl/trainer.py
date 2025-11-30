"""
Simple Policy Gradient Trainer for LLM Society Survival

This module provides a standalone trainer that can:
1. Train a simple policy network using REINFORCE
2. Compare trained policies against baselines (random, heuristic)
3. Produce behavioral fingerprints of different policies

No external RL library required - just numpy.

Usage:
    python -m llm_society.rl.trainer --episodes 500 --eval-every 50
"""

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from llm_society.rl.atropos_env import SurvivalWorld, SurvivalWorldConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Policy Networks (Simple NumPy Implementation)
# =============================================================================

@dataclass
class PolicyConfig:
    """Configuration for the policy network."""
    state_dim: int = 8  # energy, health, food, pos_x, pos_y, step, connections, nearby
    action_dim: int = 4  # rest, gather, move, talk
    hidden_dim: int = 32
    learning_rate: float = 0.01
    gamma: float = 0.99  # Discount factor
    entropy_coef: float = 0.01  # Entropy regularization


class SimplePolicy:
    """
    Simple 2-layer MLP policy using NumPy.

    Architecture: state -> hidden -> action_logits -> softmax -> action_probs
    """

    def __init__(self, config: PolicyConfig):
        self.config = config

        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(config.state_dim, config.hidden_dim) * np.sqrt(2.0 / config.state_dim)
        self.b1 = np.zeros(config.hidden_dim)
        self.W2 = np.random.randn(config.hidden_dim, config.action_dim) * np.sqrt(2.0 / config.hidden_dim)
        self.b2 = np.zeros(config.action_dim)

        # For tracking gradients
        self.saved_log_probs = []
        self.rewards = []

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / exp_x.sum()

    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass: state -> action probabilities."""
        h = self._relu(state @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        probs = self._softmax(logits)
        return probs

    def select_action(self, state: np.ndarray) -> Tuple[int, float]:
        """Select action and return (action_idx, log_prob)."""
        probs = self.forward(state)
        action = np.random.choice(len(probs), p=probs)
        log_prob = np.log(probs[action] + 1e-10)
        return action, log_prob

    def get_params(self) -> np.ndarray:
        """Flatten all parameters into a single vector."""
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten(),
        ])

    def set_params(self, params: np.ndarray):
        """Set parameters from a flattened vector."""
        idx = 0
        W1_size = self.config.state_dim * self.config.hidden_dim
        b1_size = self.config.hidden_dim
        W2_size = self.config.hidden_dim * self.config.action_dim
        b2_size = self.config.action_dim

        self.W1 = params[idx:idx + W1_size].reshape(self.config.state_dim, self.config.hidden_dim)
        idx += W1_size
        self.b1 = params[idx:idx + b1_size]
        idx += b1_size
        self.W2 = params[idx:idx + W2_size].reshape(self.config.hidden_dim, self.config.action_dim)
        idx += W2_size
        self.b2 = params[idx:idx + b2_size]

    def compute_loss_and_update(self) -> float:
        """Clear buffers and return total reward (for compatibility)."""
        total_reward = sum(self.rewards) if self.rewards else 0.0
        self.saved_log_probs = []
        self.rewards = []
        return total_reward

    def save(self, path: str):
        """Save policy weights."""
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path: str):
        """Load policy weights."""
        data = np.load(path)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']


# =============================================================================
# State/Action Conversion
# =============================================================================

ACTION_NAMES = ["rest", "gather_resources", "move_to", "talk_to"]

# State schema for trajectory logging (maps indices to feature names)
STATE_SCHEMA = {
    0: "energy",           # 0.0-1.0, dies if reaches 0
    1: "health",           # 0.0-1.0, dies if reaches 0
    2: "food_normalized",  # food/10, raw food consumed 1 every 10 steps
    3: "position_x_norm",  # x/20, world is 20x20
    4: "position_y_norm",  # y/20
    5: "step_fraction",    # step/200, episode ends at 200
    6: "connections_norm", # num_connections/5, clamped to 1.0
    7: "nearby_norm",      # nearby_agent_count/5, clamped to 1.0
}


def obs_to_state(obs: Dict[str, Any]) -> np.ndarray:
    """Convert observation dict to state vector."""
    return np.array([
        obs.get("energy", 0.0),
        obs.get("health", 0.0),
        obs.get("food", 0) / 10.0,  # Normalize
        obs.get("position_x", 0.0) / 20.0,  # Normalize
        obs.get("position_y", 0.0) / 20.0,  # Normalize
        obs.get("step", 0) / 200.0,  # Normalize
        min(1.0, obs.get("num_connections", 0) / 5.0),  # Normalize
        min(1.0, len(obs.get("nearby_agents", [])) / 5.0),  # Normalize
    ], dtype=np.float32)


def action_to_dict(action_idx: int, obs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert action index to action dict."""
    action_type = ACTION_NAMES[action_idx]

    if action_type == "move_to":
        # Random direction for now
        return {
            "type": "move_to",
            "params": {
                "x": random.uniform(0, obs.get("world_width", 20)),
                "y": random.uniform(0, obs.get("world_height", 20)),
            }
        }
    elif action_type == "talk_to":
        nearby = obs.get("nearby_agents", [])
        if nearby:
            target = random.choice(nearby)
            return {"type": "talk_to", "params": {"target_id": target["id"]}}
        else:
            return {"type": "rest", "params": {}}
    else:
        return {"type": action_type, "params": {}}


# =============================================================================
# Baseline Policies
# =============================================================================

def random_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Random policy."""
    action_idx = random.randint(0, 2)  # rest, gather, move
    return action_to_dict(action_idx, obs)


def heuristic_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Survival heuristic policy."""
    if obs["energy"] < 0.3:
        return {"type": "rest", "params": {}}
    if obs["food"] <= 2:
        return {"type": "gather_resources", "params": {}}
    return {"type": "rest", "params": {}}


# =============================================================================
# Training Loop
# =============================================================================

@dataclass
class TrainingMetrics:
    """Metrics from training."""
    episode: int = 0
    total_reward: float = 0.0
    episode_length: int = 0
    survived: bool = False
    action_counts: Dict[str, int] = field(default_factory=dict)
    final_health: float = 0.0
    final_food: int = 0
    final_energy: float = 0.0
    trajectory: List[Dict[str, Any]] = field(default_factory=list)  # Step-by-step log
    seed: int = 0


def run_episode(
    world: SurvivalWorld,
    policy: SimplePolicy,
    agent_id: str = "agent_0",
    training: bool = True,
    log_trajectory: bool = False,
) -> TrainingMetrics:
    """Run a single episode and return metrics."""
    obs = world.reset()

    metrics = TrainingMetrics()
    metrics.action_counts = {name: 0 for name in ACTION_NAMES}
    metrics.seed = world.config.seed

    done = False
    step = 0
    while not done:
        state = obs_to_state(obs)
        action_idx, log_prob = policy.select_action(state)
        action = action_to_dict(action_idx, obs)
        action_type = action.get("type", "rest")

        metrics.action_counts[action_type] = metrics.action_counts.get(action_type, 0) + 1

        # Log trajectory step before taking action
        if log_trajectory:
            metrics.trajectory.append({
                "step": step,
                "state": state.tolist(),
                "action": action_type,
                "obs": {
                    "energy": obs["energy"],
                    "health": obs["health"],
                    "food": obs["food"],
                    "position_x": obs.get("position_x", 0),
                    "position_y": obs.get("position_y", 0),
                },
            })

        obs, reward, done, info = world.step(agent_id, action)

        # Update trajectory with reward after action
        if log_trajectory and metrics.trajectory:
            metrics.trajectory[-1]["reward"] = reward
            metrics.trajectory[-1]["done"] = done

        if training:
            policy.saved_log_probs.append(log_prob)
            policy.rewards.append(reward)

        metrics.total_reward += reward
        metrics.episode_length += 1
        step += 1

    metrics.survived = info.get("survived", False)
    metrics.final_health = obs["health"]
    metrics.final_food = obs["food"]
    metrics.final_energy = obs["energy"]

    return metrics


def evaluate_policy(
    world: SurvivalWorld,
    policy: SimplePolicy,
    num_episodes: int = 20,
    log_trajectory: bool = False,
) -> Dict[str, Any]:
    """Evaluate policy over multiple episodes."""
    results = {
        "avg_reward": 0.0,
        "survival_rate": 0.0,
        "avg_length": 0.0,
        "action_dist": {name: 0.0 for name in ACTION_NAMES},
    }

    rewards = []
    survival = []
    lengths = []
    action_totals = {name: 0 for name in ACTION_NAMES}
    episodes = []  # For trajectory logging

    for i in range(num_episodes):
        world.config.seed = 10000 + i
        metrics = run_episode(world, policy, training=False, log_trajectory=log_trajectory)

        rewards.append(metrics.total_reward)
        survival.append(1.0 if metrics.survived else 0.0)
        lengths.append(metrics.episode_length)

        for action, count in metrics.action_counts.items():
            action_totals[action] += count

        if log_trajectory:
            episodes.append({
                "episode_index": i,
                "seed": metrics.seed,
                "trajectory": metrics.trajectory,
                "total_reward": metrics.total_reward,
                "survived": metrics.survived,
                "episode_length": metrics.episode_length,
                "final_state": {
                    "health": metrics.final_health,
                    "energy": metrics.final_energy,
                    "food": metrics.final_food,
                },
                "action_counts": metrics.action_counts,
            })

    total_actions = sum(action_totals.values())

    results["avg_reward"] = sum(rewards) / len(rewards)
    results["survival_rate"] = sum(survival) / len(survival)
    results["avg_length"] = sum(lengths) / len(lengths)
    results["action_dist"] = {
        name: count / total_actions if total_actions > 0 else 0
        for name, count in action_totals.items()
    }

    if log_trajectory:
        results["episodes"] = episodes

    return results


def save_evaluation_run(
    policy_name: str,
    results: Dict[str, Any],
    world_config: SurvivalWorldConfig,
    output_dir: str = "./runs/survivalworld",
) -> str:
    """
    Save a complete evaluation run with trajectories to disk.

    Returns the path to the saved file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_policy_{policy_name}.json"
    filepath = Path(output_dir) / filename

    # Build the complete evaluation record
    record = {
        "policy": policy_name,
        "timestamp": datetime.now().isoformat(),
        "state_schema": STATE_SCHEMA,
        "action_names": ACTION_NAMES,
        "world_config": {
            "world_size": list(world_config.world_size),
            "num_agents": world_config.num_agents,
            "max_steps": world_config.max_steps,
            "initial_food": world_config.initial_food,
            "energy_decay_per_step": world_config.energy_decay_per_step,
            "rest_energy_gain": world_config.rest_energy_gain,
            "move_energy_cost": world_config.move_energy_cost,
            "gather_energy_cost": world_config.gather_energy_cost,
            "talk_energy_cost": world_config.talk_energy_cost,
            "food_consumption_interval": world_config.food_consumption_interval,
            "starvation_health_penalty": world_config.starvation_health_penalty,
        },
        "summary": {
            "avg_reward": results["avg_reward"],
            "survival_rate": results["survival_rate"],
            "avg_length": results["avg_length"],
            "action_dist": results["action_dist"],
            "num_episodes": len(results.get("episodes", [])),
        },
        "episodes": results.get("episodes", []),
    }

    with open(filepath, "w") as f:
        json.dump(record, f, indent=2)

    logger.info(f"Saved evaluation run to: {filepath}")
    return str(filepath)


def run_v2_episode(
    world: SurvivalWorld,
    policy_a,
    policy_b,
    policy_a_name: str = "agent_0",
    policy_b_name: str = "agent_1",
    log_trajectory: bool = False,
) -> Tuple[Dict[str, TrainingMetrics], Dict[str, Any]]:
    """
    Run a 2-agent episode and return per-agent metrics.

    Both agents act in the same world. We step them in alternating fashion
    but since the world increments step on each agent action, we need to be
    careful about timing. For v2, we simply step both agents each "round".

    Returns:
        (per_agent_metrics, shared_info)
    """
    obs = world.reset()

    # Initialize metrics for both agents
    metrics = {
        "agent_0": TrainingMetrics(),
        "agent_1": TrainingMetrics(),
    }
    for agent_id in metrics:
        metrics[agent_id].action_counts = {name: 0 for name in ACTION_NAMES}
        metrics[agent_id].seed = world.config.seed

    # Track per-agent done status
    done = {"agent_0": False, "agent_1": False}
    agent_obs = {"agent_0": obs, "agent_1": world._get_observation("agent_1")}

    step = 0
    max_steps = world.config.max_steps

    while step < max_steps and not all(done.values()):
        # Both agents act this round
        for agent_id, policy in [("agent_0", policy_a), ("agent_1", policy_b)]:
            if done[agent_id]:
                continue

            obs = agent_obs[agent_id]
            state = obs_to_state(obs)

            # Check if other agent is nearby
            other_agent_nearby = len(obs.get("nearby_agents", [])) > 0

            # Select action (v2 policies accept other_agent_nearby)
            if hasattr(policy, 'v2_mode') and policy.v2_mode:
                action_idx, log_prob = policy.select_action(state, other_agent_nearby)
            else:
                action_idx, log_prob = policy.select_action(state)

            action = action_to_dict(action_idx, obs)
            action_type = action.get("type", "rest")

            metrics[agent_id].action_counts[action_type] = metrics[agent_id].action_counts.get(action_type, 0) + 1

            # Log trajectory step
            if log_trajectory:
                metrics[agent_id].trajectory.append({
                    "step": step,
                    "state": state.tolist(),
                    "action": action_type,
                    "obs": {
                        "energy": obs["energy"],
                        "health": obs["health"],
                        "food": obs["food"],
                        "position_x": obs.get("position_x", 0),
                        "position_y": obs.get("position_y", 0),
                        "nearby_agents": len(obs.get("nearby_agents", [])),
                    },
                })

            # Execute action (note: world.step increments current_step)
            # We need to handle this carefully for 2 agents
            new_obs, reward, agent_done, info = world.step(agent_id, action)

            # Update trajectory with reward
            if log_trajectory and metrics[agent_id].trajectory:
                metrics[agent_id].trajectory[-1]["reward"] = reward
                metrics[agent_id].trajectory[-1]["done"] = agent_done

            metrics[agent_id].total_reward += reward
            metrics[agent_id].episode_length += 1

            if agent_done:
                done[agent_id] = True
                metrics[agent_id].survived = info.get("survived", False)
                metrics[agent_id].final_health = new_obs["health"]
                metrics[agent_id].final_food = new_obs["food"]
                metrics[agent_id].final_energy = new_obs["energy"]
            else:
                agent_obs[agent_id] = new_obs

        step += 1

    # For agents that survived, record final state
    for agent_id in ["agent_0", "agent_1"]:
        if not done[agent_id]:
            obs = agent_obs[agent_id]
            metrics[agent_id].survived = True
            metrics[agent_id].final_health = obs["health"]
            metrics[agent_id].final_food = obs["food"]
            metrics[agent_id].final_energy = obs["energy"]

    shared_info = {
        "total_steps": step,
        "both_survived": all(m.survived for m in metrics.values()),
    }

    return metrics, shared_info


def evaluate_v2_policy(
    world: SurvivalWorld,
    policy_a,
    policy_b,
    num_episodes: int = 50,
    log_trajectory: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate two policies in 2-agent world.

    Returns per-agent results plus shared metrics.
    """
    results = {
        "agent_0": {
            "avg_reward": 0.0,
            "survival_rate": 0.0,
            "avg_length": 0.0,
            "action_dist": {name: 0.0 for name in ACTION_NAMES},
        },
        "agent_1": {
            "avg_reward": 0.0,
            "survival_rate": 0.0,
            "avg_length": 0.0,
            "action_dist": {name: 0.0 for name in ACTION_NAMES},
        },
        "shared": {
            "both_survived_rate": 0.0,
        },
    }

    per_agent_rewards = {"agent_0": [], "agent_1": []}
    per_agent_survival = {"agent_0": [], "agent_1": []}
    per_agent_lengths = {"agent_0": [], "agent_1": []}
    per_agent_action_totals = {
        "agent_0": {name: 0 for name in ACTION_NAMES},
        "agent_1": {name: 0 for name in ACTION_NAMES},
    }
    both_survived = []
    episodes = {"agent_0": [], "agent_1": []}

    for i in range(num_episodes):
        world.config.seed = 10000 + i
        metrics, shared = run_v2_episode(world, policy_a, policy_b, log_trajectory=log_trajectory)

        both_survived.append(1.0 if shared["both_survived"] else 0.0)

        for agent_id in ["agent_0", "agent_1"]:
            m = metrics[agent_id]
            per_agent_rewards[agent_id].append(m.total_reward)
            per_agent_survival[agent_id].append(1.0 if m.survived else 0.0)
            per_agent_lengths[agent_id].append(m.episode_length)

            for action, count in m.action_counts.items():
                per_agent_action_totals[agent_id][action] += count

            if log_trajectory:
                episodes[agent_id].append({
                    "episode_index": i,
                    "seed": m.seed,
                    "trajectory": m.trajectory,
                    "total_reward": m.total_reward,
                    "survived": m.survived,
                    "episode_length": m.episode_length,
                    "final_state": {
                        "health": m.final_health,
                        "energy": m.final_energy,
                        "food": m.final_food,
                    },
                    "action_counts": m.action_counts,
                })

    # Compute averages
    for agent_id in ["agent_0", "agent_1"]:
        rewards = per_agent_rewards[agent_id]
        survival = per_agent_survival[agent_id]
        lengths = per_agent_lengths[agent_id]
        action_totals = per_agent_action_totals[agent_id]
        total_actions = sum(action_totals.values())

        results[agent_id]["avg_reward"] = sum(rewards) / len(rewards)
        results[agent_id]["survival_rate"] = sum(survival) / len(survival)
        results[agent_id]["avg_length"] = sum(lengths) / len(lengths)
        results[agent_id]["action_dist"] = {
            name: count / total_actions if total_actions > 0 else 0
            for name, count in action_totals.items()
        }
        if log_trajectory:
            results[agent_id]["episodes"] = episodes[agent_id]

    results["shared"]["both_survived_rate"] = sum(both_survived) / len(both_survived)

    return results


def save_v2_evaluation_run(
    policy_a_name: str,
    policy_b_name: str,
    results: Dict[str, Any],
    world_config: SurvivalWorldConfig,
    output_dir: str = "./runs/survivalworld_v2",
) -> str:
    """Save v2 evaluation run with per-agent trajectories."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_v2_{policy_a_name}_vs_{policy_b_name}.json"
    filepath = Path(output_dir) / filename

    record = {
        "version": "v2",
        "policy_a": policy_a_name,
        "policy_b": policy_b_name,
        "timestamp": datetime.now().isoformat(),
        "state_schema": STATE_SCHEMA,
        "action_names": ACTION_NAMES,
        "world_config": {
            "world_size": list(world_config.world_size),
            "num_agents": world_config.num_agents,
            "max_steps": world_config.max_steps,
            "initial_food": world_config.initial_food,
            "energy_decay_per_step": world_config.energy_decay_per_step,
            "rest_energy_gain": world_config.rest_energy_gain,
            "move_energy_cost": world_config.move_energy_cost,
            "gather_energy_cost": world_config.gather_energy_cost,
            "talk_energy_cost": world_config.talk_energy_cost,
            "food_consumption_interval": world_config.food_consumption_interval,
            "starvation_health_penalty": world_config.starvation_health_penalty,
        },
        "summary": {
            "agent_0": {
                "avg_reward": results["agent_0"]["avg_reward"],
                "survival_rate": results["agent_0"]["survival_rate"],
                "avg_length": results["agent_0"]["avg_length"],
                "action_dist": results["agent_0"]["action_dist"],
            },
            "agent_1": {
                "avg_reward": results["agent_1"]["avg_reward"],
                "survival_rate": results["agent_1"]["survival_rate"],
                "avg_length": results["agent_1"]["avg_length"],
                "action_dist": results["agent_1"]["action_dist"],
            },
            "both_survived_rate": results["shared"]["both_survived_rate"],
            "num_episodes": len(results["agent_0"].get("episodes", [])),
        },
        "agent_0_episodes": results["agent_0"].get("episodes", []),
        "agent_1_episodes": results["agent_1"].get("episodes", []),
    }

    with open(filepath, "w") as f:
        json.dump(record, f, indent=2)

    logger.info(f"Saved v2 evaluation run to: {filepath}")
    return str(filepath)


def run_v3_episode(
    world: SurvivalWorld,
    policy_a,
    policy_b,
    policy_a_name: str = "agent_0",
    policy_b_name: str = "agent_1",
    log_trajectory: bool = False,
) -> Tuple[Dict[str, TrainingMetrics], Dict[str, Any]]:
    """
    Run a v3 (2-agent with social bonus) episode and return per-agent metrics.

    Same as v2 but tracks additional social metrics:
    - Steps near other agent (within social radius)
    - Total social bonus earned
    - Average distance to other agent

    Returns:
        (per_agent_metrics, shared_info)
    """
    obs = world.reset()

    # Initialize metrics for both agents
    metrics = {
        "agent_0": TrainingMetrics(),
        "agent_1": TrainingMetrics(),
    }
    for agent_id in metrics:
        metrics[agent_id].action_counts = {name: 0 for name in ACTION_NAMES}
        metrics[agent_id].seed = world.config.seed

    # v3: Track social metrics
    social_metrics = {
        "agent_0": {"steps_near": 0, "social_bonus_total": 0.0, "distances": []},
        "agent_1": {"steps_near": 0, "social_bonus_total": 0.0, "distances": []},
    }

    # Track per-agent done status
    done = {"agent_0": False, "agent_1": False}
    agent_obs = {"agent_0": obs, "agent_1": world._get_observation("agent_1")}

    step = 0
    max_steps = world.config.max_steps

    while step < max_steps and not all(done.values()):
        # Calculate distance between agents (for tracking)
        if "agent_0" in world.agents and "agent_1" in world.agents:
            dist = world.agents["agent_0"].position.distance_to(
                world.agents["agent_1"].position
            )
        else:
            dist = float('inf')

        # Both agents act this round
        for agent_id, policy in [("agent_0", policy_a), ("agent_1", policy_b)]:
            if done[agent_id]:
                continue

            obs = agent_obs[agent_id]
            state = obs_to_state(obs)

            # Check if other agent is nearby
            other_agent_nearby = len(obs.get("nearby_agents", [])) > 0

            # Select action (v2/v3 policies accept other_agent_nearby)
            if hasattr(policy, 'v2_mode') and policy.v2_mode:
                action_idx, log_prob = policy.select_action(state, other_agent_nearby)
            else:
                action_idx, log_prob = policy.select_action(state)

            action = action_to_dict(action_idx, obs)
            action_type = action.get("type", "rest")

            metrics[agent_id].action_counts[action_type] = metrics[agent_id].action_counts.get(action_type, 0) + 1

            # Log trajectory step
            if log_trajectory:
                metrics[agent_id].trajectory.append({
                    "step": step,
                    "state": state.tolist(),
                    "action": action_type,
                    "obs": {
                        "energy": obs["energy"],
                        "health": obs["health"],
                        "food": obs["food"],
                        "position_x": obs.get("position_x", 0),
                        "position_y": obs.get("position_y", 0),
                        "nearby_agents": len(obs.get("nearby_agents", [])),
                    },
                    "distance_to_other": dist,
                })

            # Execute action
            new_obs, reward, agent_done, info = world.step(agent_id, action)

            # Track social metrics
            social_bonus = info.get("social_bonus", 0.0)
            is_near = info.get("is_near_other", False)
            social_metrics[agent_id]["social_bonus_total"] += social_bonus
            if is_near:
                social_metrics[agent_id]["steps_near"] += 1
            social_metrics[agent_id]["distances"].append(dist)

            # Update trajectory with reward and social info
            if log_trajectory and metrics[agent_id].trajectory:
                metrics[agent_id].trajectory[-1]["reward"] = reward
                metrics[agent_id].trajectory[-1]["done"] = agent_done
                metrics[agent_id].trajectory[-1]["social_bonus"] = social_bonus
                metrics[agent_id].trajectory[-1]["is_near_other"] = is_near

            metrics[agent_id].total_reward += reward
            metrics[agent_id].episode_length += 1

            if agent_done:
                done[agent_id] = True
                metrics[agent_id].survived = info.get("survived", False)
                metrics[agent_id].final_health = new_obs["health"]
                metrics[agent_id].final_food = new_obs["food"]
                metrics[agent_id].final_energy = new_obs["energy"]
            else:
                agent_obs[agent_id] = new_obs

        step += 1

    # For agents that survived, record final state
    for agent_id in ["agent_0", "agent_1"]:
        if not done[agent_id]:
            obs = agent_obs[agent_id]
            metrics[agent_id].survived = True
            metrics[agent_id].final_health = obs["health"]
            metrics[agent_id].final_food = obs["food"]
            metrics[agent_id].final_energy = obs["energy"]

    # Compute average distance
    for agent_id in social_metrics:
        dists = social_metrics[agent_id]["distances"]
        social_metrics[agent_id]["avg_distance"] = sum(dists) / len(dists) if dists else 0.0

    shared_info = {
        "total_steps": step,
        "both_survived": all(m.survived for m in metrics.values()),
        "social_metrics": social_metrics,
    }

    return metrics, shared_info


def evaluate_v3_policy(
    world: SurvivalWorld,
    policy_a,
    policy_b,
    num_episodes: int = 50,
    log_trajectory: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate two policies in v3 (2-agent world with social bonus).

    Returns per-agent results plus shared metrics including social stats.
    """
    results = {
        "agent_0": {
            "avg_reward": 0.0,
            "survival_rate": 0.0,
            "avg_length": 0.0,
            "action_dist": {name: 0.0 for name in ACTION_NAMES},
            "avg_social_bonus": 0.0,
            "social_uptime": 0.0,  # % of steps near other agent
            "avg_distance": 0.0,
        },
        "agent_1": {
            "avg_reward": 0.0,
            "survival_rate": 0.0,
            "avg_length": 0.0,
            "action_dist": {name: 0.0 for name in ACTION_NAMES},
            "avg_social_bonus": 0.0,
            "social_uptime": 0.0,
            "avg_distance": 0.0,
        },
        "shared": {
            "both_survived_rate": 0.0,
        },
    }

    per_agent_rewards = {"agent_0": [], "agent_1": []}
    per_agent_survival = {"agent_0": [], "agent_1": []}
    per_agent_lengths = {"agent_0": [], "agent_1": []}
    per_agent_action_totals = {
        "agent_0": {name: 0 for name in ACTION_NAMES},
        "agent_1": {name: 0 for name in ACTION_NAMES},
    }
    per_agent_social_bonus = {"agent_0": [], "agent_1": []}
    per_agent_steps_near = {"agent_0": [], "agent_1": []}
    per_agent_avg_distance = {"agent_0": [], "agent_1": []}

    both_survived = []
    episodes = {"agent_0": [], "agent_1": []}

    for i in range(num_episodes):
        world.config.seed = 10000 + i
        metrics, shared = run_v3_episode(world, policy_a, policy_b, log_trajectory=log_trajectory)

        both_survived.append(1.0 if shared["both_survived"] else 0.0)
        social_metrics = shared["social_metrics"]

        for agent_id in ["agent_0", "agent_1"]:
            m = metrics[agent_id]
            per_agent_rewards[agent_id].append(m.total_reward)
            per_agent_survival[agent_id].append(1.0 if m.survived else 0.0)
            per_agent_lengths[agent_id].append(m.episode_length)

            for action, count in m.action_counts.items():
                per_agent_action_totals[agent_id][action] += count

            # Social metrics
            sm = social_metrics[agent_id]
            per_agent_social_bonus[agent_id].append(sm["social_bonus_total"])
            per_agent_steps_near[agent_id].append(sm["steps_near"])
            per_agent_avg_distance[agent_id].append(sm["avg_distance"])

            if log_trajectory:
                episodes[agent_id].append({
                    "episode_index": i,
                    "seed": m.seed,
                    "trajectory": m.trajectory,
                    "total_reward": m.total_reward,
                    "survived": m.survived,
                    "episode_length": m.episode_length,
                    "final_state": {
                        "health": m.final_health,
                        "energy": m.final_energy,
                        "food": m.final_food,
                    },
                    "action_counts": m.action_counts,
                    "social_bonus_total": sm["social_bonus_total"],
                    "steps_near": sm["steps_near"],
                    "avg_distance": sm["avg_distance"],
                })

    # Compute averages
    for agent_id in ["agent_0", "agent_1"]:
        rewards = per_agent_rewards[agent_id]
        survival = per_agent_survival[agent_id]
        lengths = per_agent_lengths[agent_id]
        action_totals = per_agent_action_totals[agent_id]
        total_actions = sum(action_totals.values())

        results[agent_id]["avg_reward"] = sum(rewards) / len(rewards)
        results[agent_id]["survival_rate"] = sum(survival) / len(survival)
        results[agent_id]["avg_length"] = sum(lengths) / len(lengths)
        results[agent_id]["action_dist"] = {
            name: count / total_actions if total_actions > 0 else 0
            for name, count in action_totals.items()
        }

        # Social metrics averages
        results[agent_id]["avg_social_bonus"] = sum(per_agent_social_bonus[agent_id]) / len(per_agent_social_bonus[agent_id])
        total_steps = sum(lengths)
        total_near = sum(per_agent_steps_near[agent_id])
        results[agent_id]["social_uptime"] = total_near / total_steps if total_steps > 0 else 0.0
        results[agent_id]["avg_distance"] = sum(per_agent_avg_distance[agent_id]) / len(per_agent_avg_distance[agent_id])

        if log_trajectory:
            results[agent_id]["episodes"] = episodes[agent_id]

    results["shared"]["both_survived_rate"] = sum(both_survived) / len(both_survived)

    return results


def save_v3_evaluation_run(
    policy_a_name: str,
    policy_b_name: str,
    results: Dict[str, Any],
    world_config: SurvivalWorldConfig,
    output_dir: str = "./runs/survivalworld_v3",
) -> str:
    """Save v3 evaluation run with per-agent trajectories and social metrics."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_v3_{policy_a_name}_vs_{policy_b_name}.json"
    filepath = Path(output_dir) / filename

    record = {
        "version": "v3",
        "policy_a": policy_a_name,
        "policy_b": policy_b_name,
        "timestamp": datetime.now().isoformat(),
        "state_schema": STATE_SCHEMA,
        "action_names": ACTION_NAMES,
        "world_config": {
            "world_size": list(world_config.world_size),
            "num_agents": world_config.num_agents,
            "max_steps": world_config.max_steps,
            "initial_food": world_config.initial_food,
            "energy_decay_per_step": world_config.energy_decay_per_step,
            "rest_energy_gain": world_config.rest_energy_gain,
            "move_energy_cost": world_config.move_energy_cost,
            "gather_energy_cost": world_config.gather_energy_cost,
            "talk_energy_cost": world_config.talk_energy_cost,
            "food_consumption_interval": world_config.food_consumption_interval,
            "starvation_health_penalty": world_config.starvation_health_penalty,
            "social_radius": world_config.social_radius,
            "social_bonus": world_config.social_bonus,
            "enable_social_bonus": world_config.enable_social_bonus,
        },
        "summary": {
            "agent_0": {
                "avg_reward": results["agent_0"]["avg_reward"],
                "survival_rate": results["agent_0"]["survival_rate"],
                "avg_length": results["agent_0"]["avg_length"],
                "action_dist": results["agent_0"]["action_dist"],
                "avg_social_bonus": results["agent_0"]["avg_social_bonus"],
                "social_uptime": results["agent_0"]["social_uptime"],
                "avg_distance": results["agent_0"]["avg_distance"],
            },
            "agent_1": {
                "avg_reward": results["agent_1"]["avg_reward"],
                "survival_rate": results["agent_1"]["survival_rate"],
                "avg_length": results["agent_1"]["avg_length"],
                "action_dist": results["agent_1"]["action_dist"],
                "avg_social_bonus": results["agent_1"]["avg_social_bonus"],
                "social_uptime": results["agent_1"]["social_uptime"],
                "avg_distance": results["agent_1"]["avg_distance"],
            },
            "both_survived_rate": results["shared"]["both_survived_rate"],
            "num_episodes": len(results["agent_0"].get("episodes", [])),
        },
        "agent_0_episodes": results["agent_0"].get("episodes", []),
        "agent_1_episodes": results["agent_1"].get("episodes", []),
    }

    with open(filepath, "w") as f:
        json.dump(record, f, indent=2)

    logger.info(f"Saved v3 evaluation run to: {filepath}")
    return str(filepath)


def train_cem_v3(
    num_generations: int = 50,
    population_size: int = 20,
    elite_frac: float = 0.2,
    eval_every: int = 10,
    output_dir: str = "./training_results_v3",
    verbose: bool = False,
):
    """
    Train using Cross-Entropy Method (CEM) in v3 environment (with social bonus).

    This trains RL to exploit the proximity reward.
    """
    logger.info(f"Starting CEM v3 training (with social bonus) for {num_generations} generations...")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create v3 world (2 agents, social bonus enabled)
    world_config = SurvivalWorldConfig(
        world_size=(20.0, 20.0),
        num_agents=2,
        max_steps=200,
        initial_food=5,
        enable_social_bonus=True,  # v3!
        social_bonus=0.005,
    )
    world = SurvivalWorld(world_config)

    policy_config = PolicyConfig()
    policy = SimplePolicy(policy_config)

    # Get parameter dimensionality
    num_params = len(policy.get_params())
    logger.info(f"Policy has {num_params} parameters")

    # Initialize CEM distribution
    mean = np.zeros(num_params)
    std = np.ones(num_params) * 0.5

    num_elites = max(1, int(population_size * elite_frac))

    # Training history
    history = {
        "generations": [],
        "best_rewards": [],
        "mean_rewards": [],
        "survival_rates": [],
        "social_uptimes": [],
        "eval_results": [],
    }

    best_params = None
    best_reward = -float('inf')

    for gen in range(num_generations):
        # Sample population
        population = []
        for _ in range(population_size):
            params = mean + std * np.random.randn(num_params)
            population.append(params)

        # Evaluate each candidate (both agents use same policy)
        rewards = []
        survivals = []
        social_uptimes = []

        for i, params in enumerate(population):
            policy.set_params(params)
            policy.v2_mode = False  # RL doesn't need v2 prompts

            # Run multiple episodes for more stable evaluation
            ep_rewards = []
            ep_survivals = []
            ep_social_uptimes = []

            for seed in range(3):  # 3 episodes per candidate
                world.config.seed = gen * 1000 + i * 10 + seed
                metrics, shared = run_v3_episode(world, policy, policy)

                # Combine both agent rewards
                total_reward = (metrics["agent_0"].total_reward + metrics["agent_1"].total_reward) / 2
                both_survived = 1.0 if shared["both_survived"] else 0.0

                # Social uptime from both agents
                sm = shared["social_metrics"]
                total_steps = metrics["agent_0"].episode_length + metrics["agent_1"].episode_length
                total_near = sm["agent_0"]["steps_near"] + sm["agent_1"]["steps_near"]
                uptime = total_near / total_steps if total_steps > 0 else 0.0

                ep_rewards.append(total_reward)
                ep_survivals.append(both_survived)
                ep_social_uptimes.append(uptime)

            avg_reward = sum(ep_rewards) / len(ep_rewards)
            avg_survival = sum(ep_survivals) / len(ep_survivals)
            avg_uptime = sum(ep_social_uptimes) / len(ep_social_uptimes)

            rewards.append(avg_reward)
            survivals.append(avg_survival)
            social_uptimes.append(avg_uptime)

        # Select elites
        elite_indices = np.argsort(rewards)[-num_elites:]
        elite_params = [population[i] for i in elite_indices]
        elite_rewards = [rewards[i] for i in elite_indices]

        # Update distribution
        elite_array = np.array(elite_params)
        mean = elite_array.mean(axis=0)
        std = elite_array.std(axis=0) + 0.01  # Add noise floor

        # Track best
        gen_best_idx = np.argmax(rewards)
        if rewards[gen_best_idx] > best_reward:
            best_reward = rewards[gen_best_idx]
            best_params = population[gen_best_idx].copy()

        # Track metrics
        history["generations"].append(gen)
        history["best_rewards"].append(max(rewards))
        history["mean_rewards"].append(sum(rewards) / len(rewards))
        history["survival_rates"].append(sum(survivals) / len(survivals))
        history["social_uptimes"].append(sum(social_uptimes) / len(social_uptimes))

        # Logging
        if verbose or (gen + 1) % 5 == 0:
            logger.info(
                f"Gen {gen + 1:3d}: "
                f"best={max(rewards):.2f}, "
                f"mean={sum(rewards)/len(rewards):.2f}, "
                f"survival={sum(survivals)/len(survivals):.1%}, "
                f"social_uptime={sum(social_uptimes)/len(social_uptimes):.1%}"
            )

        # Periodic evaluation
        if (gen + 1) % eval_every == 0:
            logger.info("Running evaluation with best policy...")
            policy.set_params(best_params)
            eval_results = evaluate_v3_policy(world, policy, policy, num_episodes=10)
            history["eval_results"].append({
                "generation": gen + 1,
                **{f"agent_0_{k}": v for k, v in eval_results["agent_0"].items() if k != "episodes"},
                **{f"agent_1_{k}": v for k, v in eval_results["agent_1"].items() if k != "episodes"},
            })

            logger.info(
                f"Eval: avg_reward={eval_results['agent_0']['avg_reward']:.2f}, "
                f"survival={eval_results['agent_0']['survival_rate']:.1%}, "
                f"social_uptime={eval_results['agent_0']['social_uptime']:.1%}"
            )

    # Set best params for final policy
    policy.set_params(best_params)

    # Save policy
    policy_path = Path(output_dir) / "policy_v3_final.npz"
    policy.save(str(policy_path))
    logger.info(f"Policy saved to: {policy_path}")

    # Save training history
    history_path = Path(output_dir) / f"training_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=str)
    logger.info(f"Training history saved to: {history_path}")

    return policy, history, world_config, policy_config


def run_v3_benchmark(
    output_dir: str = "./runs/survivalworld_v3",
    num_episodes: int = 10,
    rl_v3_policy_path: Optional[str] = None,
    rl_v2_policy_path: Optional[str] = None,
    include_gemini: bool = False,
    include_groq: bool = False,
):
    """
    Run v3 (2-agent with social bonus) benchmark.

    Tests: RL-v3 vs RL-v3, RL-v2 vs RL-v2, Gemini vs Gemini, Groq vs Groq.
    Compares behavior with and without social bonus awareness.
    """
    logger.info("=" * 60)
    logger.info("SURVIVALWORLD V3 BENCHMARK (SOCIAL BONUS)")
    logger.info("=" * 60)

    # Create v3 world (2 agents, social bonus enabled)
    world_config = SurvivalWorldConfig(
        world_size=(20.0, 20.0),
        num_agents=2,
        max_steps=200,
        initial_food=5,
        enable_social_bonus=True,  # v3!
        social_bonus=0.005,
    )
    world = SurvivalWorld(world_config)

    results_summary = {}

    # Create policies
    policies = {}

    # Heuristic policy (unchanged from v2 - doesn't seek social)
    class HeuristicPolicyWrapper:
        v2_mode = False
        def select_action(self, state, other_agent_nearby=False):
            energy = state[0]
            food = state[2] * 10
            if energy < 0.3:
                return 0, 0.0  # rest
            if food <= 2:
                return 1, 0.0  # gather
            return 0, 0.0  # rest
        saved_log_probs = []
        rewards = []

    policies["heuristic"] = HeuristicPolicyWrapper()

    # RL policy trained on v3 (social aware)
    if rl_v3_policy_path and Path(rl_v3_policy_path).exists():
        policy_config = PolicyConfig()
        rl_v3_policy = SimplePolicy(policy_config)
        rl_v3_policy.load(rl_v3_policy_path)
        rl_v3_policy.v2_mode = False
        policies["rl_v3"] = rl_v3_policy
        logger.info(f"Loaded RL v3 policy from {rl_v3_policy_path}")

    # RL policy trained on v2 (not social aware, for comparison)
    if rl_v2_policy_path and Path(rl_v2_policy_path).exists():
        policy_config = PolicyConfig()
        rl_v2_policy = SimplePolicy(policy_config)
        rl_v2_policy.load(rl_v2_policy_path)
        rl_v2_policy.v2_mode = False
        policies["rl_v2"] = rl_v2_policy
        logger.info(f"Loaded RL v2 policy from {rl_v2_policy_path}")

    # Gemini policy
    if include_gemini:
        try:
            from llm_society.rl.gemini_policy import GeminiPolicy
            policies["gemini"] = GeminiPolicy(temperature=0.0, v2_mode=True)
        except Exception as e:
            logger.warning(f"Could not load Gemini policy: {e}")

    # Groq policy
    if include_groq:
        try:
            from llm_society.rl.groq_policy import GroqPolicy
            policies["groq"] = GroqPolicy(temperature=0.0, v2_mode=True)
        except Exception as e:
            logger.warning(f"Could not load Groq policy: {e}")

    # Run same-policy matchups
    logger.info("\n--- Same-Policy Matchups ---")
    for policy_name, policy in policies.items():
        logger.info(f"\nEvaluating {policy_name} vs {policy_name}...")

        results = evaluate_v3_policy(
            world, policy, policy,
            num_episodes=num_episodes,
            log_trajectory=True
        )
        save_v3_evaluation_run(policy_name, policy_name, results, world_config, output_dir)

        results_summary[f"{policy_name}_vs_{policy_name}"] = {
            "agent_0": {
                "avg_reward": results["agent_0"]["avg_reward"],
                "survival_rate": results["agent_0"]["survival_rate"],
                "action_dist": results["agent_0"]["action_dist"],
                "social_uptime": results["agent_0"]["social_uptime"],
                "avg_distance": results["agent_0"]["avg_distance"],
                "avg_social_bonus": results["agent_0"]["avg_social_bonus"],
            },
            "agent_1": {
                "avg_reward": results["agent_1"]["avg_reward"],
                "survival_rate": results["agent_1"]["survival_rate"],
                "action_dist": results["agent_1"]["action_dist"],
                "social_uptime": results["agent_1"]["social_uptime"],
                "avg_distance": results["agent_1"]["avg_distance"],
                "avg_social_bonus": results["agent_1"]["avg_social_bonus"],
            },
            "both_survived_rate": results["shared"]["both_survived_rate"],
        }

        # Get LLM stats if applicable
        if hasattr(policy, 'get_stats'):
            logger.info(f"  API stats: {policy.get_stats()}")
            policy.reset_stats()

    # Print results table
    logger.info("\n" + "=" * 100)
    logger.info("V3 BENCHMARK RESULTS (SOCIAL BONUS ENABLED)")
    logger.info("=" * 100)
    logger.info(f"{'Matchup':<20} {'Agent':<8} {'Reward':>8} {'Surv%':>7} {'%Rest':>7} {'%Gather':>8} {'%Move':>7} {'SocUp%':>8} {'AvgDist':>8}")
    logger.info("-" * 100)

    for matchup, data in results_summary.items():
        for agent_id in ["agent_0", "agent_1"]:
            agent_data = data[agent_id]
            logger.info(
                f"{matchup:<20} {agent_id:<8} "
                f"{agent_data['avg_reward']:>8.2f} "
                f"{agent_data['survival_rate']*100:>6.0f}% "
                f"{agent_data['action_dist'].get('rest', 0)*100:>6.1f}% "
                f"{agent_data['action_dist'].get('gather_resources', 0)*100:>7.1f}% "
                f"{agent_data['action_dist'].get('move_to', 0)*100:>6.1f}% "
                f"{agent_data['social_uptime']*100:>7.1f}% "
                f"{agent_data['avg_distance']:>8.1f}"
            )

    logger.info("=" * 100)
    logger.info(f"Trajectory logs saved to: {output_dir}")

    return results_summary


def run_v4_episode(
    world: SurvivalWorld,
    policy_a,
    policy_b,
    policy_a_name: str = "agent_0_policy",
    policy_b_name: str = "agent_1_policy",
    log_trajectory: bool = False,
) -> Tuple[Dict[str, TrainingMetrics], Dict[str, Any]]:
    """
    Run a v4 (mixed policy) episode and return per-agent metrics.

    V4 is the same physics as v3, but allows DIFFERENT policies per agent.
    This enables cross-policy matchups like RL v3 vs Gemini.

    Additional v4 metrics tracked:
    - who_approached_first: Which agent first moved toward the other
    - pursuit_index: (moves that shorten distance) / (total moves) per agent
    - distance_over_time: Full trajectory for visualization

    Returns:
        (per_agent_metrics, shared_info)
    """
    obs = world.reset()

    # Initialize metrics for both agents
    metrics = {
        "agent_0": TrainingMetrics(),
        "agent_1": TrainingMetrics(),
    }
    for agent_id in metrics:
        metrics[agent_id].action_counts = {name: 0 for name in ACTION_NAMES}
        metrics[agent_id].seed = world.config.seed

    # v4: Track social and interaction metrics
    social_metrics = {
        "agent_0": {"steps_near": 0, "social_bonus_total": 0.0, "distances": []},
        "agent_1": {"steps_near": 0, "social_bonus_total": 0.0, "distances": []},
    }

    # v4: Interaction metrics
    interaction_metrics = {
        "agent_0": {"moves_toward": 0, "moves_away": 0, "total_moves": 0},
        "agent_1": {"moves_toward": 0, "moves_away": 0, "total_moves": 0},
    }
    who_approached_first = None  # Will be set to "agent_0" or "agent_1"
    first_approach_step = None

    # Track per-agent done status
    done = {"agent_0": False, "agent_1": False}
    agent_obs = {"agent_0": obs, "agent_1": world._get_observation("agent_1")}

    # Store previous positions for pursuit/avoid calculation
    prev_positions = {
        "agent_0": (obs.get("position_x", 0), obs.get("position_y", 0)),
        "agent_1": (
            agent_obs["agent_1"].get("position_x", 0),
            agent_obs["agent_1"].get("position_y", 0),
        ),
    }

    step = 0
    max_steps = world.config.max_steps

    while step < max_steps and not all(done.values()):
        # Calculate distance between agents (for tracking)
        if "agent_0" in world.agents and "agent_1" in world.agents:
            dist = world.agents["agent_0"].position.distance_to(
                world.agents["agent_1"].position
            )
        else:
            dist = float('inf')

        prev_dist = dist  # Store for pursuit/avoid calculation

        # Both agents act this round
        for agent_id, policy in [("agent_0", policy_a), ("agent_1", policy_b)]:
            if done[agent_id]:
                continue

            obs = agent_obs[agent_id]
            state = obs_to_state(obs)

            # Check if other agent is nearby
            other_agent_nearby = len(obs.get("nearby_agents", [])) > 0

            # Select action (v2/v3 policies accept other_agent_nearby)
            if hasattr(policy, 'v2_mode') and policy.v2_mode:
                action_idx, log_prob = policy.select_action(state, other_agent_nearby)
            else:
                action_idx, log_prob = policy.select_action(state)

            action = action_to_dict(action_idx, obs)
            action_type = action.get("type", "rest")

            metrics[agent_id].action_counts[action_type] = metrics[agent_id].action_counts.get(action_type, 0) + 1

            # Log trajectory step
            if log_trajectory:
                metrics[agent_id].trajectory.append({
                    "step": step,
                    "state": state.tolist(),
                    "action": action_type,
                    "obs": {
                        "energy": obs["energy"],
                        "health": obs["health"],
                        "food": obs["food"],
                        "position_x": obs.get("position_x", 0),
                        "position_y": obs.get("position_y", 0),
                        "nearby_agents": len(obs.get("nearby_agents", [])),
                    },
                    "distance_to_other": dist,
                })

            # Execute action
            new_obs, reward, agent_done, info = world.step(agent_id, action)

            # Track pursuit/avoid for move actions
            if action_type == "move_to":
                interaction_metrics[agent_id]["total_moves"] += 1

                # Calculate new distance after move
                if "agent_0" in world.agents and "agent_1" in world.agents:
                    new_dist = world.agents["agent_0"].position.distance_to(
                        world.agents["agent_1"].position
                    )
                else:
                    new_dist = prev_dist

                if new_dist < prev_dist:
                    interaction_metrics[agent_id]["moves_toward"] += 1
                    # Track who approached first
                    if who_approached_first is None:
                        who_approached_first = agent_id
                        first_approach_step = step
                elif new_dist > prev_dist:
                    interaction_metrics[agent_id]["moves_away"] += 1

                prev_dist = new_dist

            # Track social metrics
            social_bonus = info.get("social_bonus", 0.0)
            is_near = info.get("is_near_other", False)
            social_metrics[agent_id]["social_bonus_total"] += social_bonus
            if is_near:
                social_metrics[agent_id]["steps_near"] += 1
            social_metrics[agent_id]["distances"].append(dist)

            # Update trajectory with reward and social info
            if log_trajectory and metrics[agent_id].trajectory:
                metrics[agent_id].trajectory[-1]["reward"] = reward
                metrics[agent_id].trajectory[-1]["done"] = agent_done
                metrics[agent_id].trajectory[-1]["social_bonus"] = social_bonus
                metrics[agent_id].trajectory[-1]["is_near_other"] = is_near

            metrics[agent_id].total_reward += reward
            metrics[agent_id].episode_length += 1

            if agent_done:
                done[agent_id] = True
                metrics[agent_id].survived = info.get("survived", False)
                metrics[agent_id].final_health = new_obs["health"]
                metrics[agent_id].final_food = new_obs["food"]
                metrics[agent_id].final_energy = new_obs["energy"]
            else:
                agent_obs[agent_id] = new_obs

        step += 1

    # For agents that survived, record final state
    for agent_id in ["agent_0", "agent_1"]:
        if not done[agent_id]:
            obs = agent_obs[agent_id]
            metrics[agent_id].survived = True
            metrics[agent_id].final_health = obs["health"]
            metrics[agent_id].final_food = obs["food"]
            metrics[agent_id].final_energy = obs["energy"]

    # Compute derived metrics
    for agent_id in social_metrics:
        dists = social_metrics[agent_id]["distances"]
        social_metrics[agent_id]["avg_distance"] = sum(dists) / len(dists) if dists else 0.0

        # Compute pursuit index: moves_toward / total_moves
        im = interaction_metrics[agent_id]
        if im["total_moves"] > 0:
            im["pursuit_index"] = im["moves_toward"] / im["total_moves"]
            im["avoid_index"] = im["moves_away"] / im["total_moves"]
        else:
            im["pursuit_index"] = 0.0
            im["avoid_index"] = 0.0

    shared_info = {
        "total_steps": step,
        "both_survived": all(m.survived for m in metrics.values()),
        "social_metrics": social_metrics,
        "interaction_metrics": interaction_metrics,
        "who_approached_first": who_approached_first,
        "first_approach_step": first_approach_step,
    }

    return metrics, shared_info


def evaluate_v4_policy(
    world: SurvivalWorld,
    policy_a,
    policy_b,
    policy_a_name: str,
    policy_b_name: str,
    num_episodes: int = 50,
    log_trajectory: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate two DIFFERENT policies in v4 (mixed policy world).

    Returns per-agent results plus shared metrics including interaction stats.
    """
    results = {
        "agent_0": {
            "policy_name": policy_a_name,
            "avg_reward": 0.0,
            "survival_rate": 0.0,
            "avg_length": 0.0,
            "action_dist": {name: 0.0 for name in ACTION_NAMES},
            "avg_social_bonus": 0.0,
            "social_uptime": 0.0,
            "avg_distance": 0.0,
            "pursuit_index": 0.0,
            "avoid_index": 0.0,
        },
        "agent_1": {
            "policy_name": policy_b_name,
            "avg_reward": 0.0,
            "survival_rate": 0.0,
            "avg_length": 0.0,
            "action_dist": {name: 0.0 for name in ACTION_NAMES},
            "avg_social_bonus": 0.0,
            "social_uptime": 0.0,
            "avg_distance": 0.0,
            "pursuit_index": 0.0,
            "avoid_index": 0.0,
        },
        "shared": {
            "both_survived_rate": 0.0,
            "who_approached_first_counts": {"agent_0": 0, "agent_1": 0, "neither": 0},
        },
    }

    per_agent_rewards = {"agent_0": [], "agent_1": []}
    per_agent_survival = {"agent_0": [], "agent_1": []}
    per_agent_lengths = {"agent_0": [], "agent_1": []}
    per_agent_action_totals = {
        "agent_0": {name: 0 for name in ACTION_NAMES},
        "agent_1": {name: 0 for name in ACTION_NAMES},
    }
    per_agent_social_bonus = {"agent_0": [], "agent_1": []}
    per_agent_steps_near = {"agent_0": [], "agent_1": []}
    per_agent_avg_distance = {"agent_0": [], "agent_1": []}
    per_agent_pursuit_index = {"agent_0": [], "agent_1": []}
    per_agent_avoid_index = {"agent_0": [], "agent_1": []}

    both_survived = []
    who_approached_counts = {"agent_0": 0, "agent_1": 0, "neither": 0}
    episodes = {"agent_0": [], "agent_1": []}

    for i in range(num_episodes):
        world.config.seed = 10000 + i
        metrics, shared = run_v4_episode(
            world, policy_a, policy_b,
            policy_a_name, policy_b_name,
            log_trajectory=log_trajectory
        )

        both_survived.append(1.0 if shared["both_survived"] else 0.0)
        social_metrics = shared["social_metrics"]
        interaction_metrics = shared["interaction_metrics"]

        # Track who approached first
        waf = shared["who_approached_first"]
        if waf:
            who_approached_counts[waf] += 1
        else:
            who_approached_counts["neither"] += 1

        for agent_id in ["agent_0", "agent_1"]:
            m = metrics[agent_id]
            per_agent_rewards[agent_id].append(m.total_reward)
            per_agent_survival[agent_id].append(1.0 if m.survived else 0.0)
            per_agent_lengths[agent_id].append(m.episode_length)

            for action, count in m.action_counts.items():
                per_agent_action_totals[agent_id][action] += count

            # Social metrics
            sm = social_metrics[agent_id]
            per_agent_social_bonus[agent_id].append(sm["social_bonus_total"])
            per_agent_steps_near[agent_id].append(sm["steps_near"])
            per_agent_avg_distance[agent_id].append(sm["avg_distance"])

            # Interaction metrics
            im = interaction_metrics[agent_id]
            per_agent_pursuit_index[agent_id].append(im["pursuit_index"])
            per_agent_avoid_index[agent_id].append(im["avoid_index"])

            if log_trajectory:
                episodes[agent_id].append({
                    "episode_index": i,
                    "seed": m.seed,
                    "trajectory": m.trajectory,
                    "total_reward": m.total_reward,
                    "survived": m.survived,
                    "episode_length": m.episode_length,
                    "final_state": {
                        "health": m.final_health,
                        "energy": m.final_energy,
                        "food": m.final_food,
                    },
                    "action_counts": m.action_counts,
                    "social_bonus_total": sm["social_bonus_total"],
                    "steps_near": sm["steps_near"],
                    "avg_distance": sm["avg_distance"],
                    "pursuit_index": im["pursuit_index"],
                    "avoid_index": im["avoid_index"],
                })

    # Compute averages
    for agent_id in ["agent_0", "agent_1"]:
        rewards = per_agent_rewards[agent_id]
        survival = per_agent_survival[agent_id]
        lengths = per_agent_lengths[agent_id]
        action_totals = per_agent_action_totals[agent_id]
        total_actions = sum(action_totals.values())

        results[agent_id]["avg_reward"] = sum(rewards) / len(rewards)
        results[agent_id]["survival_rate"] = sum(survival) / len(survival)
        results[agent_id]["avg_length"] = sum(lengths) / len(lengths)
        results[agent_id]["action_dist"] = {
            name: count / total_actions if total_actions > 0 else 0
            for name, count in action_totals.items()
        }

        # Social metrics averages
        results[agent_id]["avg_social_bonus"] = sum(per_agent_social_bonus[agent_id]) / len(per_agent_social_bonus[agent_id])
        total_steps = sum(lengths)
        total_near = sum(per_agent_steps_near[agent_id])
        results[agent_id]["social_uptime"] = total_near / total_steps if total_steps > 0 else 0.0
        results[agent_id]["avg_distance"] = sum(per_agent_avg_distance[agent_id]) / len(per_agent_avg_distance[agent_id])

        # Interaction metrics averages
        results[agent_id]["pursuit_index"] = sum(per_agent_pursuit_index[agent_id]) / len(per_agent_pursuit_index[agent_id])
        results[agent_id]["avoid_index"] = sum(per_agent_avoid_index[agent_id]) / len(per_agent_avoid_index[agent_id])

        if log_trajectory:
            results[agent_id]["episodes"] = episodes[agent_id]

    results["shared"]["both_survived_rate"] = sum(both_survived) / len(both_survived)
    results["shared"]["who_approached_first_counts"] = who_approached_counts

    return results


def save_v4_evaluation_run(
    policy_a_name: str,
    policy_b_name: str,
    results: Dict[str, Any],
    world_config: SurvivalWorldConfig,
    output_dir: str = "./runs/survivalworld_v4",
) -> str:
    """Save v4 evaluation run with per-agent trajectories and interaction metrics."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_v4_{policy_a_name}_vs_{policy_b_name}.json"
    filepath = Path(output_dir) / filename

    record = {
        "version": "v4",
        "policy_a": policy_a_name,
        "policy_b": policy_b_name,
        "timestamp": datetime.now().isoformat(),
        "state_schema": STATE_SCHEMA,
        "action_names": ACTION_NAMES,
        "world_config": {
            "world_size": list(world_config.world_size),
            "num_agents": world_config.num_agents,
            "max_steps": world_config.max_steps,
            "initial_food": world_config.initial_food,
            "energy_decay_per_step": world_config.energy_decay_per_step,
            "rest_energy_gain": world_config.rest_energy_gain,
            "move_energy_cost": world_config.move_energy_cost,
            "gather_energy_cost": world_config.gather_energy_cost,
            "talk_energy_cost": world_config.talk_energy_cost,
            "food_consumption_interval": world_config.food_consumption_interval,
            "starvation_health_penalty": world_config.starvation_health_penalty,
            "social_radius": world_config.social_radius,
            "social_bonus": world_config.social_bonus,
            "enable_social_bonus": world_config.enable_social_bonus,
        },
        "summary": {
            "agent_0": {
                "policy_name": results["agent_0"]["policy_name"],
                "avg_reward": results["agent_0"]["avg_reward"],
                "survival_rate": results["agent_0"]["survival_rate"],
                "avg_length": results["agent_0"]["avg_length"],
                "action_dist": results["agent_0"]["action_dist"],
                "avg_social_bonus": results["agent_0"]["avg_social_bonus"],
                "social_uptime": results["agent_0"]["social_uptime"],
                "avg_distance": results["agent_0"]["avg_distance"],
                "pursuit_index": results["agent_0"]["pursuit_index"],
                "avoid_index": results["agent_0"]["avoid_index"],
            },
            "agent_1": {
                "policy_name": results["agent_1"]["policy_name"],
                "avg_reward": results["agent_1"]["avg_reward"],
                "survival_rate": results["agent_1"]["survival_rate"],
                "avg_length": results["agent_1"]["avg_length"],
                "action_dist": results["agent_1"]["action_dist"],
                "avg_social_bonus": results["agent_1"]["avg_social_bonus"],
                "social_uptime": results["agent_1"]["social_uptime"],
                "avg_distance": results["agent_1"]["avg_distance"],
                "pursuit_index": results["agent_1"]["pursuit_index"],
                "avoid_index": results["agent_1"]["avoid_index"],
            },
            "both_survived_rate": results["shared"]["both_survived_rate"],
            "who_approached_first_counts": results["shared"]["who_approached_first_counts"],
            "num_episodes": len(results["agent_0"].get("episodes", [])),
        },
        "agent_0_episodes": results["agent_0"].get("episodes", []),
        "agent_1_episodes": results["agent_1"].get("episodes", []),
    }

    with open(filepath, "w") as f:
        json.dump(record, f, indent=2)

    logger.info(f"Saved v4 evaluation run to: {filepath}")
    return str(filepath)


def run_v4_benchmark(
    output_dir: str = "./runs/survivalworld_v4",
    num_episodes: int = 10,
    rl_v3_policy_path: Optional[str] = None,
    include_gemini: bool = True,
    include_groq: bool = True,
):
    """
    Run v4 (mixed policy) benchmark.

    V4 answers: What happens when DIFFERENT kinds of minds share the same world?

    Mixed pairings:
    - RL v3 vs Gemini
    - RL v3 vs Groq
    - Gemini vs Groq

    Also runs symmetric baselines for comparison:
    - RL v3 vs RL v3
    - Gemini vs Gemini
    - Groq vs Groq
    """
    logger.info("=" * 60)
    logger.info("SURVIVALWORLD V4 BENCHMARK (MIXED POLICY)")
    logger.info("=" * 60)
    logger.info("Question: What happens when different minds share the same world?")
    logger.info("=" * 60)

    # Create v4 world (same physics as v3)
    world_config = SurvivalWorldConfig(
        world_size=(20.0, 20.0),
        num_agents=2,
        max_steps=200,
        initial_food=5,
        enable_social_bonus=True,  # Same as v3
        social_bonus=0.005,
    )
    world = SurvivalWorld(world_config)

    results_summary = {}
    policies = {}

    # Load RL v3 policy
    if rl_v3_policy_path and Path(rl_v3_policy_path).exists():
        policy_config = PolicyConfig()
        rl_v3_policy = SimplePolicy(policy_config)
        rl_v3_policy.load(rl_v3_policy_path)
        rl_v3_policy.v2_mode = False
        policies["rl_v3"] = rl_v3_policy
        logger.info(f"Loaded RL v3 policy from {rl_v3_policy_path}")
    else:
        logger.warning(f"RL v3 policy not found at {rl_v3_policy_path}")

    # Load Gemini policy
    if include_gemini:
        try:
            from llm_society.rl.gemini_policy import GeminiPolicy
            policies["gemini"] = GeminiPolicy(temperature=0.0, v2_mode=True)
            logger.info("Loaded Gemini policy")
        except Exception as e:
            logger.warning(f"Could not load Gemini policy: {e}")

    # Load Groq policy
    if include_groq:
        try:
            from llm_society.rl.groq_policy import GroqPolicy
            policies["groq"] = GroqPolicy(temperature=0.0, v2_mode=True)
            logger.info("Loaded Groq policy")
        except Exception as e:
            logger.warning(f"Could not load Groq policy: {e}")

    # Define matchups: (policy_a_name, policy_b_name)
    # First run symmetric baselines
    symmetric_matchups = []
    for name in policies.keys():
        symmetric_matchups.append((name, name))

    # Then mixed pairings
    mixed_matchups = []
    policy_names = list(policies.keys())
    for i, name_a in enumerate(policy_names):
        for name_b in policy_names[i+1:]:
            mixed_matchups.append((name_a, name_b))

    # Run symmetric baselines first
    logger.info("\n" + "=" * 60)
    logger.info("SYMMETRIC BASELINES")
    logger.info("=" * 60)

    for policy_a_name, policy_b_name in symmetric_matchups:
        logger.info(f"\nRunning {policy_a_name} vs {policy_b_name}...")

        policy_a = policies[policy_a_name]
        policy_b = policies[policy_b_name]

        results = evaluate_v4_policy(
            world, policy_a, policy_b,
            policy_a_name, policy_b_name,
            num_episodes=num_episodes,
            log_trajectory=True
        )
        save_v4_evaluation_run(policy_a_name, policy_b_name, results, world_config, output_dir)

        matchup_key = f"{policy_a_name}_vs_{policy_b_name}"
        results_summary[matchup_key] = {
            "type": "symmetric",
            "agent_0": {
                "policy": policy_a_name,
                "avg_reward": results["agent_0"]["avg_reward"],
                "survival_rate": results["agent_0"]["survival_rate"],
                "action_dist": results["agent_0"]["action_dist"],
                "social_uptime": results["agent_0"]["social_uptime"],
                "avg_distance": results["agent_0"]["avg_distance"],
                "pursuit_index": results["agent_0"]["pursuit_index"],
            },
            "agent_1": {
                "policy": policy_b_name,
                "avg_reward": results["agent_1"]["avg_reward"],
                "survival_rate": results["agent_1"]["survival_rate"],
                "action_dist": results["agent_1"]["action_dist"],
                "social_uptime": results["agent_1"]["social_uptime"],
                "avg_distance": results["agent_1"]["avg_distance"],
                "pursuit_index": results["agent_1"]["pursuit_index"],
            },
            "both_survived_rate": results["shared"]["both_survived_rate"],
            "who_approached_first": results["shared"]["who_approached_first_counts"],
        }

        # Reset LLM stats if applicable
        if hasattr(policy_a, 'get_stats'):
            logger.info(f"  {policy_a_name} API stats: {policy_a.get_stats()}")
            policy_a.reset_stats()

    # Run mixed pairings
    logger.info("\n" + "=" * 60)
    logger.info("MIXED PAIRINGS (The main event!)")
    logger.info("=" * 60)

    for policy_a_name, policy_b_name in mixed_matchups:
        logger.info(f"\nRunning {policy_a_name} vs {policy_b_name}...")

        policy_a = policies[policy_a_name]
        policy_b = policies[policy_b_name]

        results = evaluate_v4_policy(
            world, policy_a, policy_b,
            policy_a_name, policy_b_name,
            num_episodes=num_episodes,
            log_trajectory=True
        )
        save_v4_evaluation_run(policy_a_name, policy_b_name, results, world_config, output_dir)

        matchup_key = f"{policy_a_name}_vs_{policy_b_name}"
        results_summary[matchup_key] = {
            "type": "mixed",
            "agent_0": {
                "policy": policy_a_name,
                "avg_reward": results["agent_0"]["avg_reward"],
                "survival_rate": results["agent_0"]["survival_rate"],
                "action_dist": results["agent_0"]["action_dist"],
                "social_uptime": results["agent_0"]["social_uptime"],
                "avg_distance": results["agent_0"]["avg_distance"],
                "pursuit_index": results["agent_0"]["pursuit_index"],
            },
            "agent_1": {
                "policy": policy_b_name,
                "avg_reward": results["agent_1"]["avg_reward"],
                "survival_rate": results["agent_1"]["survival_rate"],
                "action_dist": results["agent_1"]["action_dist"],
                "social_uptime": results["agent_1"]["social_uptime"],
                "avg_distance": results["agent_1"]["avg_distance"],
                "pursuit_index": results["agent_1"]["pursuit_index"],
            },
            "both_survived_rate": results["shared"]["both_survived_rate"],
            "who_approached_first": results["shared"]["who_approached_first_counts"],
        }

        # Reset LLM stats
        for p_name, p in [(policy_a_name, policy_a), (policy_b_name, policy_b)]:
            if hasattr(p, 'get_stats'):
                logger.info(f"  {p_name} API stats: {p.get_stats()}")
                p.reset_stats()

    # Print comprehensive results
    logger.info("\n" + "=" * 120)
    logger.info("V4 BENCHMARK RESULTS: MIXED POLICY MATCHUPS")
    logger.info("=" * 120)

    # Table header
    header = f"{'Matchup':<25} {'Type':<10} {'Agent':<8} {'Policy':<10} {'Reward':>8} {'Surv%':>7} {'%Move':>7} {'SocUp%':>8} {'Pursuit':>8}"
    logger.info(header)
    logger.info("-" * 120)

    for matchup, data in results_summary.items():
        for agent_id in ["agent_0", "agent_1"]:
            agent_data = data[agent_id]
            if agent_id == "agent_0":
                matchup_str = matchup
                type_str = data["type"]
            else:
                matchup_str = ""
                type_str = ""

            logger.info(
                f"{matchup_str:<25} {type_str:<10} {agent_id:<8} "
                f"{agent_data['policy']:<10} "
                f"{agent_data['avg_reward']:>8.2f} "
                f"{agent_data['survival_rate']*100:>6.0f}% "
                f"{agent_data['action_dist'].get('move_to', 0)*100:>6.1f}% "
                f"{agent_data['social_uptime']*100:>7.1f}% "
                f"{agent_data['pursuit_index']*100:>7.1f}%"
            )

    # Print "who approached first" summary for mixed matchups
    logger.info("\n" + "=" * 80)
    logger.info("WHO APPROACHED FIRST (Mixed Matchups Only)")
    logger.info("=" * 80)
    for matchup, data in results_summary.items():
        if data["type"] == "mixed":
            waf = data["who_approached_first"]
            logger.info(f"{matchup}: agent_0={waf['agent_0']}, agent_1={waf['agent_1']}, neither={waf['neither']}")

    logger.info("\n" + "=" * 80)
    logger.info(f"All trajectory logs saved to: {output_dir}")
    logger.info("=" * 80)

    return results_summary


def run_v2_benchmark(
    output_dir: str = "./runs/survivalworld_v2",
    num_episodes: int = 50,
    rl_policy_path: Optional[str] = None,
    include_gemini: bool = False,
    include_groq: bool = False,
):
    """
    Run v2 (2-agent) benchmark.

    Tests: RL vs RL, Gemini vs Gemini, Groq vs Groq, and cross-policy matchups.
    """
    logger.info("=" * 60)
    logger.info("SURVIVALWORLD V2 BENCHMARK (2-AGENT)")
    logger.info("=" * 60)

    # Create 2-agent world
    world_config = SurvivalWorldConfig(
        world_size=(20.0, 20.0),
        num_agents=2,  # v2 change!
        max_steps=200,
        initial_food=5,
    )
    world = SurvivalWorld(world_config)

    results_summary = {}
    matchups_run = []

    # Create policies
    policies = {}

    # Heuristic policy
    class HeuristicPolicyWrapper:
        v2_mode = False
        def select_action(self, state, other_agent_nearby=False):
            energy = state[0]
            food = state[2] * 10
            if energy < 0.3:
                return 0, 0.0  # rest
            if food <= 2:
                return 1, 0.0  # gather
            return 0, 0.0  # rest
        saved_log_probs = []
        rewards = []

    policies["heuristic"] = HeuristicPolicyWrapper()

    # RL policy
    if rl_policy_path and Path(rl_policy_path).exists():
        policy_config = PolicyConfig()
        rl_policy = SimplePolicy(policy_config)
        rl_policy.load(rl_policy_path)
        rl_policy.v2_mode = False  # RL doesn't use presence info
        policies["rl"] = rl_policy

    # Gemini policy
    if include_gemini:
        try:
            from llm_society.rl.gemini_policy import GeminiPolicy
            policies["gemini"] = GeminiPolicy(temperature=0.0, v2_mode=True)
        except Exception as e:
            logger.warning(f"Could not load Gemini policy: {e}")

    # Groq policy
    if include_groq:
        try:
            from llm_society.rl.groq_policy import GroqPolicy
            policies["groq"] = GroqPolicy(temperature=0.0, v2_mode=True)
        except Exception as e:
            logger.warning(f"Could not load Groq policy: {e}")

    # Run same-policy matchups first (e.g., RL vs RL)
    logger.info("\n--- Same-Policy Matchups ---")
    for policy_name, policy in policies.items():
        logger.info(f"\nEvaluating {policy_name} vs {policy_name}...")

        results = evaluate_v2_policy(
            world, policy, policy,
            num_episodes=num_episodes,
            log_trajectory=True
        )
        save_v2_evaluation_run(policy_name, policy_name, results, world_config, output_dir)

        results_summary[f"{policy_name}_vs_{policy_name}"] = {
            "agent_0": {
                "avg_reward": results["agent_0"]["avg_reward"],
                "survival_rate": results["agent_0"]["survival_rate"],
                "action_dist": results["agent_0"]["action_dist"],
            },
            "agent_1": {
                "avg_reward": results["agent_1"]["avg_reward"],
                "survival_rate": results["agent_1"]["survival_rate"],
                "action_dist": results["agent_1"]["action_dist"],
            },
            "both_survived_rate": results["shared"]["both_survived_rate"],
        }
        matchups_run.append((policy_name, policy_name))

        # Get LLM stats if applicable
        if hasattr(policy, 'get_stats'):
            logger.info(f"  API stats: {policy.get_stats()}")
            policy.reset_stats()

    # Print results table
    logger.info("\n" + "=" * 80)
    logger.info("V2 BENCHMARK RESULTS")
    logger.info("=" * 80)
    logger.info(f"{'Matchup':<25} {'Agent':<8} {'Reward':>10} {'Survival':>10} {'%Rest':>8} {'%Gather':>8}")
    logger.info("-" * 80)

    for matchup, data in results_summary.items():
        for agent_id in ["agent_0", "agent_1"]:
            agent_data = data[agent_id]
            logger.info(
                f"{matchup:<25} {agent_id:<8} "
                f"{agent_data['avg_reward']:>10.2f} "
                f"{agent_data['survival_rate']*100:>9.1f}% "
                f"{agent_data['action_dist'].get('rest', 0)*100:>7.1f}% "
                f"{agent_data['action_dist'].get('gather_resources', 0)*100:>7.1f}%"
            )

    logger.info("=" * 80)
    logger.info(f"Trajectory logs saved to: {output_dir}")

    return results_summary


def run_benchmark(
    output_dir: str = "./runs/survivalworld",
    num_episodes: int = 50,
    rl_policy_path: Optional[str] = None,
    include_gemini: bool = False,
):
    """
    Run benchmark evaluation for all policies (Random, Heuristic, RL, optionally Gemini).

    Saves trajectory logs for each policy.
    """
    num_policies = 3 + (1 if include_gemini else 0)

    logger.info("=" * 60)
    logger.info("SURVIVALWORLD BENCHMARK EVALUATION")
    logger.info("=" * 60)

    # Create world
    world_config = SurvivalWorldConfig(
        world_size=(20.0, 20.0),
        num_agents=1,
        max_steps=200,
        initial_food=5,
    )
    world = SurvivalWorld(world_config)

    results_summary = {}
    policy_num = 1

    # 1. Random Policy
    logger.info(f"\n[{policy_num}/{num_policies}] Evaluating Random Policy...")
    policy_num += 1

    class RandomPolicyWrapper:
        def select_action(self, state):
            action_idx = random.randint(0, 2)  # rest, gather, move (no talk)
            return action_idx, 0.0
        saved_log_probs = []
        rewards = []

    random_policy = RandomPolicyWrapper()
    random_results = evaluate_policy(world, random_policy, num_episodes=num_episodes, log_trajectory=True)
    save_evaluation_run("random", random_results, world_config, output_dir)
    results_summary["random"] = {
        "avg_reward": random_results["avg_reward"],
        "survival_rate": random_results["survival_rate"],
        "action_dist": random_results["action_dist"],
    }

    # 2. Heuristic Policy
    logger.info(f"\n[{policy_num}/{num_policies}] Evaluating Heuristic Policy...")
    policy_num += 1

    class HeuristicPolicyWrapper:
        def select_action(self, state):
            # state[0] = energy, state[2] = food_normalized
            energy = state[0]
            food = state[2] * 10  # denormalize

            if energy < 0.3:
                return 0, 0.0  # rest
            if food <= 2:
                return 1, 0.0  # gather_resources
            return 0, 0.0  # rest

        saved_log_probs = []
        rewards = []

    heuristic_policy = HeuristicPolicyWrapper()
    heuristic_results = evaluate_policy(world, heuristic_policy, num_episodes=num_episodes, log_trajectory=True)
    save_evaluation_run("heuristic", heuristic_results, world_config, output_dir)
    results_summary["heuristic"] = {
        "avg_reward": heuristic_results["avg_reward"],
        "survival_rate": heuristic_results["survival_rate"],
        "action_dist": heuristic_results["action_dist"],
    }

    # 3. RL Policy (if available)
    if rl_policy_path and Path(rl_policy_path).exists():
        logger.info(f"\n[{policy_num}/{num_policies}] Evaluating RL Policy from {rl_policy_path}...")
        policy_num += 1
        policy_config = PolicyConfig()
        rl_policy = SimplePolicy(policy_config)
        rl_policy.load(rl_policy_path)

        rl_results = evaluate_policy(world, rl_policy, num_episodes=num_episodes, log_trajectory=True)
        save_evaluation_run("rl_cem", rl_results, world_config, output_dir)
        results_summary["rl_cem"] = {
            "avg_reward": rl_results["avg_reward"],
            "survival_rate": rl_results["survival_rate"],
            "action_dist": rl_results["action_dist"],
        }
    else:
        logger.info(f"\n[{policy_num}/{num_policies}] Skipping RL Policy (no policy file provided or found)")
        policy_num += 1

    # 4. Gemini Policy (if requested)
    if include_gemini:
        logger.info(f"\n[{policy_num}/{num_policies}] Evaluating Gemini Policy...")
        try:
            from llm_society.rl.gemini_policy import GeminiPolicy
            gemini_policy = GeminiPolicy(temperature=0.0)

            gemini_results = evaluate_policy(world, gemini_policy, num_episodes=num_episodes, log_trajectory=True)
            save_evaluation_run("gemini", gemini_results, world_config, output_dir)

            # Add Gemini stats to results
            gemini_stats = gemini_policy.get_stats()
            results_summary["gemini"] = {
                "avg_reward": gemini_results["avg_reward"],
                "survival_rate": gemini_results["survival_rate"],
                "action_dist": gemini_results["action_dist"],
                "api_stats": gemini_stats,
            }

            logger.info(f"  Gemini API stats: {gemini_stats}")

        except Exception as e:
            logger.error(f"Failed to run Gemini policy: {e}")

    # Print comparison table
    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"{'Policy':<12} {'Avg Reward':>12} {'Survival':>10} {'%Rest':>8} {'%Gather':>8} {'%Move':>8}")
    logger.info("-" * 60)

    for name, data in results_summary.items():
        logger.info(
            f"{name:<12} "
            f"{data['avg_reward']:>12.2f} "
            f"{data['survival_rate']*100:>9.1f}% "
            f"{data['action_dist'].get('rest', 0)*100:>7.1f}% "
            f"{data['action_dist'].get('gather_resources', 0)*100:>7.1f}% "
            f"{data['action_dist'].get('move_to', 0)*100:>7.1f}%"
        )

    logger.info("=" * 60)
    logger.info(f"Trajectory logs saved to: {output_dir}")

    return results_summary


def train_cem(
    num_generations: int = 50,
    population_size: int = 20,
    elite_frac: float = 0.2,
    eval_every: int = 10,
    output_dir: str = "./training_results",
    verbose: bool = False,
):
    """
    Train using Cross-Entropy Method (CEM).

    CEM is a population-based optimization that:
    1. Samples a population of parameter vectors from a Gaussian
    2. Evaluates each candidate
    3. Selects the top performers (elites)
    4. Updates the Gaussian to center on elites
    """
    logger.info(f"Starting CEM training for {num_generations} generations...")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create world and policy
    world_config = SurvivalWorldConfig(
        world_size=(20.0, 20.0),
        num_agents=1,
        max_steps=200,
        initial_food=5,
    )
    world = SurvivalWorld(world_config)

    policy_config = PolicyConfig()
    policy = SimplePolicy(policy_config)

    # Get parameter dimensionality
    num_params = len(policy.get_params())
    logger.info(f"Policy has {num_params} parameters")

    # Initialize CEM distribution
    mean = np.zeros(num_params)
    std = np.ones(num_params) * 0.5

    num_elites = max(1, int(population_size * elite_frac))

    # Training history
    history = {
        "generations": [],
        "best_rewards": [],
        "mean_rewards": [],
        "survival_rates": [],
        "eval_results": [],
    }

    best_params = None
    best_reward = -float('inf')

    for gen in range(num_generations):
        # Sample population
        population = []
        for _ in range(population_size):
            params = mean + std * np.random.randn(num_params)
            population.append(params)

        # Evaluate each candidate
        rewards = []
        survivals = []
        for i, params in enumerate(population):
            policy.set_params(params)

            # Run multiple episodes for more stable evaluation
            ep_rewards = []
            ep_survivals = []
            for seed in range(3):  # 3 episodes per candidate
                world.config.seed = gen * 1000 + i * 10 + seed
                metrics = run_episode(world, policy, training=False)
                ep_rewards.append(metrics.total_reward)
                ep_survivals.append(1.0 if metrics.survived else 0.0)

            avg_reward = sum(ep_rewards) / len(ep_rewards)
            avg_survival = sum(ep_survivals) / len(ep_survivals)
            rewards.append(avg_reward)
            survivals.append(avg_survival)

        # Select elites
        elite_indices = np.argsort(rewards)[-num_elites:]
        elite_params = [population[i] for i in elite_indices]
        elite_rewards = [rewards[i] for i in elite_indices]

        # Update distribution
        elite_array = np.array(elite_params)
        mean = elite_array.mean(axis=0)
        std = elite_array.std(axis=0) + 0.01  # Add noise floor

        # Track best
        gen_best_idx = np.argmax(rewards)
        if rewards[gen_best_idx] > best_reward:
            best_reward = rewards[gen_best_idx]
            best_params = population[gen_best_idx].copy()

        # Track metrics
        history["generations"].append(gen)
        history["best_rewards"].append(max(rewards))
        history["mean_rewards"].append(sum(rewards) / len(rewards))
        history["survival_rates"].append(sum(survivals) / len(survivals))

        # Logging
        if verbose or (gen + 1) % 5 == 0:
            logger.info(
                f"Gen {gen + 1:3d}: "
                f"best={max(rewards):.2f}, "
                f"mean={sum(rewards)/len(rewards):.2f}, "
                f"survival={sum(survivals)/len(survivals):.1%}, "
                f"elite_avg={sum(elite_rewards)/len(elite_rewards):.2f}"
            )

        # Periodic evaluation
        if (gen + 1) % eval_every == 0:
            logger.info("Running evaluation with best policy...")
            policy.set_params(best_params)
            eval_results = evaluate_policy(world, policy, num_episodes=20)
            history["eval_results"].append({
                "generation": gen + 1,
                **eval_results
            })

            logger.info(
                f"Eval: avg_reward={eval_results['avg_reward']:.2f}, "
                f"survival={eval_results['survival_rate']:.1%}"
            )

    # Set best params for final policy
    policy.set_params(best_params)
    return policy, history, world_config, policy_config


def train(
    num_episodes: int = 500,
    eval_every: int = 50,
    output_dir: str = "./training_results",
    verbose: bool = False,
):
    """Main training loop using CEM."""
    # Use CEM instead of REINFORCE
    num_generations = num_episodes // 10  # Roughly similar compute
    policy, history, world_config, policy_config = train_cem(
        num_generations=num_generations,
        population_size=20,
        elite_frac=0.2,
        eval_every=max(1, num_generations // 10),
        output_dir=output_dir,
        verbose=verbose,
    )

    world = SurvivalWorld(world_config)

    # Final evaluation
    logger.info("\n" + "=" * 50)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 50)

    final_eval = evaluate_policy(world, policy, num_episodes=50)
    logger.info(f"Trained Policy:")
    logger.info(f"  Avg Reward: {final_eval['avg_reward']:.3f}")
    logger.info(f"  Survival Rate: {final_eval['survival_rate']:.1%}")
    logger.info(f"  Avg Length: {final_eval['avg_length']:.1f}")
    logger.info(f"  Action Distribution: {final_eval['action_dist']}")

    # Compare with baselines
    logger.info("\nBaseline Comparison:")

    # Heuristic baseline
    class HeuristicWrapper:
        def select_action(self, state):
            # Reconstruct obs from state
            obs = {
                "energy": state[0],
                "health": state[1],
                "food": int(state[2] * 10),
            }
            action = heuristic_policy(obs)
            action_type = action["type"]
            action_idx = ACTION_NAMES.index(action_type) if action_type in ACTION_NAMES else 0
            return action_idx, 0.0

        saved_log_probs = []
        rewards = []

    heuristic = HeuristicWrapper()
    heuristic_eval = evaluate_policy(world, heuristic, num_episodes=50)
    logger.info(f"Heuristic Policy:")
    logger.info(f"  Avg Reward: {heuristic_eval['avg_reward']:.3f}")
    logger.info(f"  Survival Rate: {heuristic_eval['survival_rate']:.1%}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_episodes": num_episodes,
            "policy": policy_config.__dict__,
            "world": world_config.__dict__,
        },
        "training_history": history,
        "final_evaluation": final_eval,
        "heuristic_baseline": heuristic_eval,
    }

    results_path = Path(output_dir) / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_path}")

    # Save policy
    policy_path = Path(output_dir) / "policy_final.npz"
    policy.save(str(policy_path))
    logger.info(f"Policy saved to: {policy_path}")

    return policy, history, final_eval


def main():
    parser = argparse.ArgumentParser(description="Train and benchmark survival policies")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a new policy using CEM")
    train_parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    train_parser.add_argument("--eval-every", type=int, default=50, help="Evaluate every N episodes")
    train_parser.add_argument("--output", type=str, default="./training_results", help="Output directory")
    train_parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    # Benchmark subcommand
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark evaluation with trajectory logging")
    bench_parser.add_argument("--episodes", type=int, default=50, help="Number of episodes per policy")
    bench_parser.add_argument("--output", type=str, default="./runs/survivalworld", help="Output directory for logs")
    bench_parser.add_argument("--rl-policy", type=str, default="./training_results_cem/policy_final.npz",
                              help="Path to trained RL policy (optional)")
    bench_parser.add_argument("--gemini", action="store_true", help="Include Gemini policy in benchmark")

    # V2 Benchmark subcommand (2-agent)
    v2_parser = subparsers.add_parser("benchmark-v2", help="Run v2 (2-agent) benchmark")
    v2_parser.add_argument("--episodes", type=int, default=50, help="Number of episodes per matchup")
    v2_parser.add_argument("--output", type=str, default="./runs/survivalworld_v2", help="Output directory")
    v2_parser.add_argument("--rl-policy", type=str, default="./training_results_cem/policy_final.npz",
                           help="Path to trained RL policy")
    v2_parser.add_argument("--gemini", action="store_true", help="Include Gemini policy")
    v2_parser.add_argument("--groq", action="store_true", help="Include Groq policy")

    # V3 Train subcommand (2-agent with social bonus)
    train_v3_parser = subparsers.add_parser("train-v3", help="Train RL policy for v3 (with social bonus)")
    train_v3_parser.add_argument("--generations", type=int, default=50, help="Number of CEM generations")
    train_v3_parser.add_argument("--population", type=int, default=20, help="Population size")
    train_v3_parser.add_argument("--output", type=str, default="./training_results_v3", help="Output directory")
    train_v3_parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    # V3 Benchmark subcommand (2-agent with social bonus)
    v3_parser = subparsers.add_parser("benchmark-v3", help="Run v3 (2-agent, social bonus) benchmark")
    v3_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per matchup")
    v3_parser.add_argument("--output", type=str, default="./runs/survivalworld_v3", help="Output directory")
    v3_parser.add_argument("--rl-v3-policy", type=str, default="./training_results_v3/policy_v3_final.npz",
                           help="Path to RL policy trained on v3")
    v3_parser.add_argument("--rl-v2-policy", type=str, default="./training_results_cem/policy_final.npz",
                           help="Path to RL policy trained on v2 (for comparison)")
    v3_parser.add_argument("--gemini", action="store_true", help="Include Gemini policy")
    v3_parser.add_argument("--groq", action="store_true", help="Include Groq policy")

    # V4 Benchmark subcommand (mixed policy matchups)
    v4_parser = subparsers.add_parser("benchmark-v4", help="Run v4 (mixed policy) benchmark")
    v4_parser.add_argument("--episodes", type=int, default=10, help="Number of episodes per matchup")
    v4_parser.add_argument("--output", type=str, default="./runs/survivalworld_v4", help="Output directory")
    v4_parser.add_argument("--rl-v3-policy", type=str, default="./training_results_v3/policy_v3_final.npz",
                           help="Path to RL policy trained on v3")
    v4_parser.add_argument("--gemini", action="store_true", help="Include Gemini policy")
    v4_parser.add_argument("--groq", action="store_true", help="Include Groq policy")

    args = parser.parse_args()

    if args.command == "train":
        train(
            num_episodes=args.episodes,
            eval_every=args.eval_every,
            output_dir=args.output,
            verbose=args.verbose,
        )
    elif args.command == "benchmark":
        run_benchmark(
            output_dir=args.output,
            num_episodes=args.episodes,
            rl_policy_path=args.rl_policy,
            include_gemini=args.gemini,
        )
    elif args.command == "benchmark-v2":
        run_v2_benchmark(
            output_dir=args.output,
            num_episodes=args.episodes,
            rl_policy_path=args.rl_policy,
            include_gemini=args.gemini,
            include_groq=args.groq,
        )
    elif args.command == "train-v3":
        train_cem_v3(
            num_generations=args.generations,
            population_size=args.population,
            output_dir=args.output,
            verbose=args.verbose,
        )
    elif args.command == "benchmark-v3":
        run_v3_benchmark(
            output_dir=args.output,
            num_episodes=args.episodes,
            rl_v3_policy_path=args.rl_v3_policy,
            rl_v2_policy_path=args.rl_v2_policy,
            include_gemini=args.gemini,
            include_groq=args.groq,
        )
    elif args.command == "benchmark-v4":
        run_v4_benchmark(
            output_dir=args.output,
            num_episodes=args.episodes,
            rl_v3_policy_path=args.rl_v3_policy,
            include_gemini=args.gemini,
            include_groq=args.groq,
        )
    else:
        # Default: run train for backward compatibility
        train(
            num_episodes=500,
            eval_every=50,
            output_dir="./training_results",
            verbose=False,
        )


if __name__ == "__main__":
    main()
