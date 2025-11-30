"""
LLM Society Atropos Environment

This module provides an Atropos-compatible environment wrapper for the LLM Society
survival simulation. It enables RL training on the survival world.

Key design principles:
1. Environment is separate from policy - the RL trainer chooses actions
2. Simple survival reward: r_t = Δhealth + 0.1 * Δfood
3. Episode ends when agent dies or times out
4. Observations are numeric state dicts (energy, food, health, position)

Usage:
    python -m llm_society.rl.atropos_env serve --config configs/llm_society.yaml
"""

import asyncio
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

# Optional atroposlib imports - only needed for Atropos RL training
# The core SurvivalWorld can be used without atroposlib
try:
    from atroposlib.envs.base import BaseEnv, BaseEnvConfig, ScoredDataGroup
    from atroposlib.type_definitions import Item
    ATROPOS_AVAILABLE = True
except ImportError:
    BaseEnv = object
    BaseEnvConfig = object
    ScoredDataGroup = None
    Item = None
    ATROPOS_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Lightweight Survival World (no LLM, no database, pure RL)
# =============================================================================

@dataclass
class Position:
    """Simple 2D position."""
    x: float = 0.0
    y: float = 0.0

    def distance_to(self, other: "Position") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def move_towards(self, target: "Position", speed: float) -> "Position":
        dist = self.distance_to(target)
        if dist <= speed:
            return Position(target.x, target.y)
        ratio = speed / dist
        return Position(
            self.x + (target.x - self.x) * ratio,
            self.y + (target.y - self.y) * ratio,
        )


@dataclass
class Agent:
    """Lightweight agent for RL training."""
    agent_id: str
    position: Position = field(default_factory=Position)
    energy: float = 1.0
    health: float = 1.0
    food: int = 5
    currency: int = 500
    social_connections: Dict[str, float] = field(default_factory=dict)
    step_count: int = 0


@dataclass
class SurvivalWorldConfig:
    """Configuration for the lightweight survival world."""
    world_size: Tuple[float, float] = (20.0, 20.0)
    num_agents: int = 1
    max_steps: int = 200

    # Energy dynamics (v1.1)
    energy_decay_per_step: float = 0.02
    rest_energy_gain: float = 0.05
    move_energy_cost: float = 0.03
    talk_energy_cost: float = 0.02
    gather_energy_cost: float = 0.04

    # Food dynamics (v1.1)
    initial_food: int = 5
    food_consumption_interval: int = 10
    starvation_health_penalty: float = 0.1
    gather_food_min: int = 1
    gather_food_max: int = 3

    # Social
    social_radius: float = 3.0

    # v3: Social proximity bonus
    social_bonus: float = 0.005  # Reward when within social_radius of another agent
    enable_social_bonus: bool = False  # Off by default (v1/v2 mode), on for v3

    # Random seed
    seed: Optional[int] = None


class SurvivalWorld:
    """
    Lightweight survival world for RL training.

    This is a simplified version of the full LLM Society simulation,
    designed for fast RL rollouts without LLM calls or database operations.
    """

    # Valid action types
    ACTIONS = ["rest", "move_to", "gather_resources", "talk_to"]

    def __init__(self, config: SurvivalWorldConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.agents: Dict[str, Agent] = {}
        self.current_step = 0

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Reset the world and return initial observation."""
        if seed is not None:
            self.rng = random.Random(seed)

        self.current_step = 0
        self.agents = {}

        # Create agents
        for i in range(self.config.num_agents):
            agent_id = f"agent_{i}"
            self.agents[agent_id] = Agent(
                agent_id=agent_id,
                position=Position(
                    self.rng.uniform(0, self.config.world_size[0]),
                    self.rng.uniform(0, self.config.world_size[1]),
                ),
                energy=1.0,
                health=1.0,
                food=self.config.initial_food,
                currency=500,
            )

        # Return observation for the first agent (single-agent RL)
        return self._get_observation("agent_0")

    def step(self, agent_id: str, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict]:
        """
        Execute one step for an agent.

        Args:
            agent_id: The agent taking the action
            action: Dict with 'type' and optional 'params'

        Returns:
            (observation, reward, done, info)
        """
        agent = self.agents.get(agent_id)
        if agent is None:
            return {}, 0.0, True, {"error": "Agent not found"}

        # Store previous state for reward calculation
        prev_health = agent.health
        prev_food = agent.food
        prev_energy = agent.energy

        # Apply baseline energy decay
        agent.energy = max(0.0, agent.energy - self.config.energy_decay_per_step)

        # Execute action
        action_type = action.get("type", "rest")
        params = action.get("params", {})

        if action_type == "rest":
            self._execute_rest(agent)
        elif action_type == "move_to":
            self._execute_move(agent, params)
        elif action_type == "gather_resources":
            self._execute_gather(agent)
        elif action_type == "talk_to":
            self._execute_talk(agent, params)
        else:
            # Unknown action, treat as rest
            self._execute_rest(agent)

        # Food consumption
        agent.step_count += 1
        if agent.step_count % self.config.food_consumption_interval == 0:
            agent.food = max(0, agent.food - 1)
            if agent.food == 0:
                agent.health = max(0.0, agent.health - self.config.starvation_health_penalty)
                agent.energy = max(0.0, agent.energy - 0.05)

        self.current_step += 1

        # Calculate reward: survival-focused
        # Components:
        # 1. Small living bonus (+0.01 per step alive)
        # 2. Health maintenance (penalize health loss heavily)
        # 3. Energy maintenance (penalize low energy)
        # 4. Food buffer (small bonus for having food reserves)
        delta_health = agent.health - prev_health
        delta_food = agent.food - prev_food
        delta_energy = agent.energy - prev_energy

        # Living bonus
        living_bonus = 0.01

        # Health penalty (heavily penalize health loss from starvation)
        health_reward = delta_health * 2.0  # Double weight on health changes

        # Energy maintenance (penalize running low on energy)
        energy_penalty = -0.05 if agent.energy < 0.3 else 0.0

        # Food buffer (small reward for maintaining food reserves)
        food_reward = 0.02 if agent.food >= 3 else (-0.02 if agent.food == 0 else 0.0)

        # v3: Social proximity bonus
        social_bonus_reward = 0.0
        is_near_other = False
        if self.config.enable_social_bonus and self.config.num_agents > 1:
            for other_id, other in self.agents.items():
                if other_id != agent_id:
                    dist = agent.position.distance_to(other.position)
                    if dist <= self.config.social_radius:
                        social_bonus_reward = self.config.social_bonus
                        is_near_other = True
                        break  # Only need one nearby agent for the bonus

        reward = living_bonus + health_reward + energy_penalty + food_reward + social_bonus_reward

        # Check done conditions
        done = (
            agent.health <= 0 or
            agent.energy <= 0 or
            self.current_step >= self.config.max_steps
        )

        # Get observation
        obs = self._get_observation(agent_id)

        info = {
            "step": self.current_step,
            "action_type": action_type,
            "delta_health": delta_health,
            "delta_food": delta_food,
            "survived": agent.health > 0 and agent.energy > 0,
            "social_bonus": social_bonus_reward,
            "is_near_other": is_near_other,
        }

        return obs, reward, done, info

    def _execute_rest(self, agent: Agent):
        """Rest action: recover energy."""
        agent.energy = min(1.0, agent.energy + self.config.rest_energy_gain)

    def _execute_move(self, agent: Agent, params: Dict):
        """Move action: move towards target position."""
        x = params.get("x", agent.position.x)
        y = params.get("y", agent.position.y)

        # Clamp to world bounds
        x = max(0, min(self.config.world_size[0], x))
        y = max(0, min(self.config.world_size[1], y))

        target = Position(x, y)
        agent.position = agent.position.move_towards(target, 1.0)  # Speed 1.0
        agent.energy = max(0.0, agent.energy - self.config.move_energy_cost)

    def _execute_gather(self, agent: Agent):
        """Gather action: get food."""
        food_gain = self.rng.randint(
            self.config.gather_food_min,
            self.config.gather_food_max
        )
        agent.food += food_gain
        agent.energy = max(0.0, agent.energy - self.config.gather_energy_cost)

    def _execute_talk(self, agent: Agent, params: Dict):
        """Talk action: interact with another agent."""
        target_id = params.get("target_id")
        if target_id is None or target_id not in self.agents:
            return

        target = self.agents[target_id]
        distance = agent.position.distance_to(target.position)

        if distance <= self.config.social_radius:
            # Successful interaction
            if target_id not in agent.social_connections:
                agent.social_connections[target_id] = 0.3
            else:
                agent.social_connections[target_id] = min(
                    1.0, agent.social_connections[target_id] + 0.1
                )

        agent.energy = max(0.0, agent.energy - self.config.talk_energy_cost)

    def _get_observation(self, agent_id: str) -> Dict[str, Any]:
        """Get observation for an agent."""
        agent = self.agents.get(agent_id)
        if agent is None:
            return {}

        # Get nearby agents
        nearby = []
        for other_id, other in self.agents.items():
            if other_id != agent_id:
                dist = agent.position.distance_to(other.position)
                if dist <= self.config.social_radius * 2:  # Detection range
                    nearby.append({
                        "id": other_id,
                        "distance": dist,
                        "in_talk_range": dist <= self.config.social_radius,
                    })

        return {
            "energy": agent.energy,
            "health": agent.health,
            "food": agent.food,
            "currency": agent.currency,
            "position_x": agent.position.x,
            "position_y": agent.position.y,
            "world_width": self.config.world_size[0],
            "world_height": self.config.world_size[1],
            "step": self.current_step,
            "max_steps": self.config.max_steps,
            "num_connections": len(agent.social_connections),
            "nearby_agents": nearby,
        }

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)


# =============================================================================
# Atropos Environment Configuration
# =============================================================================

class LlmSocietyEnvConfig(BaseEnvConfig):
    """Configuration for the LLM Society Atropos environment."""

    # World configuration
    world_width: float = Field(default=20.0, description="World width")
    world_height: float = Field(default=20.0, description="World height")
    num_agents: int = Field(default=1, description="Number of agents (1 for single-agent RL)")
    max_episode_steps: int = Field(default=200, description="Maximum steps per episode")

    # Energy dynamics
    energy_decay: float = Field(default=0.02, description="Energy decay per step")
    rest_gain: float = Field(default=0.05, description="Energy gain from rest")
    move_cost: float = Field(default=0.03, description="Energy cost for moving")
    gather_cost: float = Field(default=0.04, description="Energy cost for gathering")
    talk_cost: float = Field(default=0.02, description="Energy cost for talking")

    # Food dynamics
    initial_food: int = Field(default=5, description="Initial food amount")
    food_interval: int = Field(default=10, description="Steps between food consumption")
    starvation_penalty: float = Field(default=0.1, description="Health penalty for starvation")

    # Social
    social_radius: float = Field(default=3.0, description="Radius for social interactions")

    # RL specific
    reward_scale: float = Field(default=1.0, description="Scale factor for rewards")


# =============================================================================
# Atropos Environment Implementation
# =============================================================================

class LlmSocietyEnv(BaseEnv):
    """
    Atropos environment for LLM Society survival simulation.

    This environment wraps the SurvivalWorld for RL training via Atropos.
    The policy (RL model or LLM) chooses actions, and this environment
    executes them and returns rewards.
    """

    name = "llm_society_survival"
    env_config_cls = LlmSocietyEnvConfig

    def __init__(self, config: LlmSocietyEnvConfig, server_configs, **kwargs):
        super().__init__(config, server_configs, **kwargs)
        self.world: Optional[SurvivalWorld] = None
        self.agent_id = "agent_0"
        self.episode_count = 0
        self.total_rewards = []
        self.survival_rates = []

    async def setup(self):
        """Initialize the environment."""
        logger.info("Setting up LLM Society Atropos Environment")

        # Create world config from env config
        world_config = SurvivalWorldConfig(
            world_size=(self.config.world_width, self.config.world_height),
            num_agents=self.config.num_agents,
            max_steps=self.config.max_episode_steps,
            energy_decay_per_step=self.config.energy_decay,
            rest_energy_gain=self.config.rest_gain,
            move_energy_cost=self.config.move_cost,
            gather_energy_cost=self.config.gather_cost,
            talk_energy_cost=self.config.talk_cost,
            initial_food=self.config.initial_food,
            food_consumption_interval=self.config.food_interval,
            starvation_health_penalty=self.config.starvation_penalty,
            social_radius=self.config.social_radius,
        )

        self.world = SurvivalWorld(world_config)
        logger.info(f"Survival world created: {world_config.world_size}")

    async def get_next_item(self) -> Item:
        """
        Get the next item for rollout.

        For LLM Society, an item represents an episode seed/configuration.
        """
        self.episode_count += 1

        # Reset world for new episode
        seed = self.episode_count  # Deterministic seed based on episode count
        obs = self.world.reset(seed=seed)

        # Create item with initial observation as context
        item = {
            "item_id": f"episode_{self.episode_count}",
            "data": {
                "episode_id": self.episode_count,
                "seed": seed,
                "initial_obs": obs,
            },
            "messages": [
                {
                    "role": "system",
                    "content": self._create_system_prompt(),
                },
                {
                    "role": "user",
                    "content": self._obs_to_prompt(obs),
                },
            ],
        }

        return item

    def _create_system_prompt(self) -> str:
        """Create the system prompt describing the environment."""
        return """You are an agent in a survival simulation.

RULES:
- You lose energy every step. At 0 energy, you cannot act.
- You lose 1 food every 10 steps. At 0 food, your health drops.
- If health reaches 0, you die.

ACTIONS (respond with just the action):
- rest: Recover some energy
- gather_resources: Get 1-3 food (costs energy)
- move_to X Y: Move towards position (X, Y)
- talk_to AGENT_ID: Talk to a nearby agent

GOAL: Survive as long as possible by managing energy and food.

Respond with ONE action only, e.g., "gather_resources" or "move_to 10 15"."""

    def _obs_to_prompt(self, obs: Dict[str, Any]) -> str:
        """Convert observation to a prompt string."""
        nearby_str = ""
        if obs.get("nearby_agents"):
            nearby = obs["nearby_agents"]
            nearby_str = f"\nNearby agents: {', '.join(a['id'] for a in nearby)}"

        return f"""Current state:
- Energy: {obs['energy']:.2f}
- Health: {obs['health']:.2f}
- Food: {obs['food']}
- Position: ({obs['position_x']:.1f}, {obs['position_y']:.1f})
- Step: {obs['step']}/{obs['max_steps']}
- Connections: {obs['num_connections']}{nearby_str}

What action do you take?"""

    def _parse_action(self, text: str) -> Dict[str, Any]:
        """Parse LLM response into action dict."""
        text = text.strip().lower()

        if text.startswith("rest"):
            return {"type": "rest", "params": {}}

        if text.startswith("gather"):
            return {"type": "gather_resources", "params": {}}

        if text.startswith("move_to"):
            parts = text.split()
            if len(parts) >= 3:
                try:
                    x = float(parts[1])
                    y = float(parts[2])
                    return {"type": "move_to", "params": {"x": x, "y": y}}
                except ValueError:
                    pass
            return {"type": "move_to", "params": {"x": 10.0, "y": 10.0}}

        if text.startswith("talk_to"):
            parts = text.split()
            if len(parts) >= 2:
                target_id = parts[1]
                return {"type": "talk_to", "params": {"target_id": target_id}}
            return {"type": "rest", "params": {}}

        # Default to rest for unparseable actions
        return {"type": "rest", "params": {}}

    async def collect_trajectory(self, item: Item) -> Tuple[Optional[Dict], List[Item]]:
        """
        Collect a single trajectory (episode) using the policy.

        This is the core RL loop:
        1. Get observation
        2. Query policy for action
        3. Execute action
        4. Accumulate reward
        5. Repeat until done
        """
        # Reset world with seed from item
        seed = item["data"]["seed"]
        obs = self.world.reset(seed=seed)

        messages = list(item["messages"])  # Copy messages

        total_reward = 0.0
        trajectory_tokens = []
        trajectory_masks = []
        step_rewards = []

        done = False
        while not done:
            # Get action from policy (LLM)
            prompt = self._obs_to_prompt(obs)
            messages.append({"role": "user", "content": prompt})

            try:
                response = await self.server.generate(
                    messages=messages,
                    max_tokens=50,
                    temperature=0.7,
                )
                action_text = response.get("content", "rest")
            except Exception as e:
                logger.warning(f"Policy query failed: {e}")
                action_text = "rest"

            messages.append({"role": "assistant", "content": action_text})

            # Parse and execute action
            action = self._parse_action(action_text)
            obs, reward, done, info = self.world.step(self.agent_id, action)

            total_reward += reward
            step_rewards.append(reward)

        # Record survival
        survived = info.get("survived", False)
        self.survival_rates.append(1.0 if survived else 0.0)
        self.total_rewards.append(total_reward)

        # Tokenize the trajectory for training
        full_text = ""
        for msg in messages:
            full_text += f"{msg['role']}: {msg['content']}\n"

        tokens = self.tokenizer.encode(full_text)
        masks = [1] * len(tokens)  # All tokens are trainable

        # Create scored data item
        scored_item = {
            "tokens": tokens,
            "masks": masks,
            "scores": total_reward * self.config.reward_scale,
            "messages": messages,
        }

        return scored_item, []

    async def evaluate(self, *args, **kwargs):
        """Evaluate the current policy."""
        logger.info("Running evaluation...")

        # Run several evaluation episodes
        eval_rewards = []
        eval_survival = []

        for i in range(10):
            obs = self.world.reset(seed=10000 + i)
            total_reward = 0.0
            done = False

            while not done:
                # Use a simple heuristic for evaluation baseline
                # or query the policy
                if obs["food"] <= 2:
                    action = {"type": "gather_resources", "params": {}}
                elif obs["energy"] < 0.3:
                    action = {"type": "rest", "params": {}}
                else:
                    action = {"type": "rest", "params": {}}

                obs, reward, done, info = self.world.step(self.agent_id, action)
                total_reward += reward

            eval_rewards.append(total_reward)
            eval_survival.append(1.0 if info.get("survived", False) else 0.0)

        avg_reward = sum(eval_rewards) / len(eval_rewards)
        avg_survival = sum(eval_survival) / len(eval_survival)

        logger.info(f"Eval - Avg Reward: {avg_reward:.3f}, Survival Rate: {avg_survival:.1%}")

        # Log to wandb
        await self.wandb_log({
            "eval/avg_reward": avg_reward,
            "eval/survival_rate": avg_survival,
            "eval/episodes": len(eval_rewards),
        })


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    LlmSocietyEnv.cli()
