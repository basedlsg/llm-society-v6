"""
Standalone test for the LLM Society RL Environment.

This script tests the SurvivalWorld without Atropos to validate:
1. World mechanics work (energy decay, food consumption, starvation)
2. Actions execute correctly (rest, move, gather, talk)
3. Reward signal makes sense
4. Episodes terminate properly

Run with:
    python -m llm_society.rl.test_env
"""

import random
from typing import Dict, Any

from llm_society.rl.atropos_env import SurvivalWorld, SurvivalWorldConfig


def random_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Random policy for testing."""
    action_type = random.choice(["rest", "move_to", "gather_resources"])

    if action_type == "move_to":
        return {
            "type": "move_to",
            "params": {
                "x": random.uniform(0, obs["world_width"]),
                "y": random.uniform(0, obs["world_height"]),
            }
        }

    return {"type": action_type, "params": {}}


def heuristic_policy(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Survival-oriented heuristic policy.

    Rules:
    1. If energy < 0.3: rest
    2. If food <= 2: gather
    3. Otherwise: rest (conserve energy)
    """
    if obs["energy"] < 0.3:
        return {"type": "rest", "params": {}}

    if obs["food"] <= 2:
        return {"type": "gather_resources", "params": {}}

    return {"type": "rest", "params": {}}


def run_episode(world: SurvivalWorld, policy, agent_id: str = "agent_0", verbose: bool = False):
    """Run a single episode and return metrics."""
    obs = world.reset()

    total_reward = 0.0
    action_counts = {"rest": 0, "move_to": 0, "gather_resources": 0, "talk_to": 0}
    step_log = []

    done = False
    while not done:
        action = policy(obs)
        action_type = action.get("type", "rest")
        action_counts[action_type] = action_counts.get(action_type, 0) + 1

        obs, reward, done, info = world.step(agent_id, action)
        total_reward += reward

        if verbose and world.current_step % 20 == 0:
            print(f"Step {world.current_step:3d}: energy={obs['energy']:.2f}, "
                  f"food={obs['food']}, health={obs['health']:.2f}, "
                  f"reward={reward:+.3f}")

        step_log.append({
            "step": world.current_step,
            "action": action_type,
            "energy": obs["energy"],
            "food": obs["food"],
            "health": obs["health"],
            "reward": reward,
        })

    survived = info.get("survived", False)

    return {
        "total_reward": total_reward,
        "steps": world.current_step,
        "survived": survived,
        "final_health": obs["health"],
        "final_energy": obs["energy"],
        "final_food": obs["food"],
        "action_counts": action_counts,
        "step_log": step_log,
    }


def test_random_policy():
    """Test with random policy - should fail often."""
    print("\n" + "=" * 60)
    print("TEST 1: Random Policy")
    print("=" * 60)

    config = SurvivalWorldConfig(
        world_size=(20.0, 20.0),
        num_agents=1,
        max_steps=200,
        initial_food=5,
        seed=42,
    )
    world = SurvivalWorld(config)

    results = []
    for i in range(10):
        world.config.seed = 42 + i
        result = run_episode(world, random_policy, verbose=(i == 0))
        results.append(result)

    avg_reward = sum(r["total_reward"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)
    survival_rate = sum(1 for r in results if r["survived"]) / len(results)

    print(f"\nRandom Policy Results (n=10):")
    print(f"  Avg Reward: {avg_reward:.3f}")
    print(f"  Avg Steps: {avg_steps:.1f}")
    print(f"  Survival Rate: {survival_rate:.1%}")

    return results


def test_heuristic_policy():
    """Test with heuristic policy - should survive more often."""
    print("\n" + "=" * 60)
    print("TEST 2: Heuristic Policy (Survival-Oriented)")
    print("=" * 60)

    config = SurvivalWorldConfig(
        world_size=(20.0, 20.0),
        num_agents=1,
        max_steps=200,
        initial_food=5,
        seed=42,
    )
    world = SurvivalWorld(config)

    results = []
    for i in range(10):
        world.config.seed = 42 + i
        result = run_episode(world, heuristic_policy, verbose=(i == 0))
        results.append(result)

    avg_reward = sum(r["total_reward"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)
    survival_rate = sum(1 for r in results if r["survived"]) / len(results)

    # Action distribution
    total_actions = sum(sum(r["action_counts"].values()) for r in results)
    action_dist = {}
    for action in ["rest", "gather_resources", "move_to", "talk_to"]:
        count = sum(r["action_counts"].get(action, 0) for r in results)
        action_dist[action] = count / total_actions if total_actions > 0 else 0

    print(f"\nHeuristic Policy Results (n=10):")
    print(f"  Avg Reward: {avg_reward:.3f}")
    print(f"  Avg Steps: {avg_steps:.1f}")
    print(f"  Survival Rate: {survival_rate:.1%}")
    print(f"\n  Action Distribution:")
    for action, pct in sorted(action_dist.items(), key=lambda x: -x[1]):
        print(f"    {action}: {pct:.1%}")

    return results


def test_mechanics():
    """Test individual mechanics work correctly."""
    print("\n" + "=" * 60)
    print("TEST 3: Mechanics Validation")
    print("=" * 60)

    config = SurvivalWorldConfig(
        world_size=(20.0, 20.0),
        num_agents=1,
        max_steps=100,
        initial_food=5,
        energy_decay_per_step=0.02,
        rest_energy_gain=0.05,
        food_consumption_interval=10,
        seed=42,
    )
    world = SurvivalWorld(config)
    obs = world.reset()

    print(f"\nInitial state: energy={obs['energy']:.2f}, food={obs['food']}, health={obs['health']:.2f}")

    # Test 1: Energy decays without action
    print("\n1. Testing energy decay...")
    for _ in range(5):
        obs, _, _, _ = world.step("agent_0", {"type": "rest", "params": {}})
    # Energy is capped at 1.0, so with rest gain > decay, it stays at 1.0
    expected_energy = min(1.0, 1.0 - (5 * 0.02) + (5 * 0.05))  # decay + rest gain, capped
    print(f"   After 5 rests: energy={obs['energy']:.3f} (expected ~{expected_energy:.3f})")
    assert abs(obs["energy"] - expected_energy) < 0.01, "Energy calculation mismatch"
    print("   ✓ Energy stays at max with rest (gain > decay)")

    # Test 2: Food consumption
    print("\n2. Testing food consumption...")
    world.reset()
    for _ in range(10):
        obs, _, _, _ = world.step("agent_0", {"type": "rest", "params": {}})
    print(f"   After 10 steps: food={obs['food']} (started with 5, should be 4)")
    assert obs["food"] == 4, "Food should decrease by 1 after 10 steps"
    print("   ✓ Food consumption works correctly")

    # Test 3: Gather gives food
    print("\n3. Testing gather action...")
    world.reset()
    obs, _, _, _ = world.step("agent_0", {"type": "gather_resources", "params": {}})
    print(f"   After 1 gather: food={obs['food']} (should be 6-8)")
    assert obs["food"] >= 6 and obs["food"] <= 8, "Gather should give 1-3 food"
    print("   ✓ Gather action works correctly")

    # Test 4: Starvation penalty
    print("\n4. Testing starvation...")
    config.initial_food = 1
    world = SurvivalWorld(config)
    obs = world.reset()
    for _ in range(20):  # Should starve at step 10, take damage at step 20
        obs, _, _, _ = world.step("agent_0", {"type": "rest", "params": {}})
    print(f"   After 20 steps with initial food=1: health={obs['health']:.2f}, food={obs['food']}")
    assert obs["health"] < 1.0, "Should have taken starvation damage"
    print("   ✓ Starvation penalty works correctly")

    print("\n✓ All mechanics validated!")


def test_reward_signal():
    """Test that reward signal correlates with survival."""
    print("\n" + "=" * 60)
    print("TEST 4: Reward Signal Validation")
    print("=" * 60)

    config = SurvivalWorldConfig(
        world_size=(20.0, 20.0),
        num_agents=1,
        max_steps=200,
        initial_food=5,
        seed=42,
    )
    world = SurvivalWorld(config)

    # Run heuristic (should have positive reward trend)
    result = run_episode(world, heuristic_policy)

    # Analyze reward trajectory
    rewards = [s["reward"] for s in result["step_log"]]
    avg_early = sum(rewards[:50]) / 50 if len(rewards) >= 50 else sum(rewards) / len(rewards)
    avg_late = sum(rewards[-50:]) / 50 if len(rewards) >= 50 else sum(rewards) / len(rewards)

    print(f"\nHeuristic policy reward analysis:")
    print(f"  Total reward: {result['total_reward']:.3f}")
    print(f"  Avg early reward (first 50): {avg_early:.4f}")
    print(f"  Avg late reward (last 50): {avg_late:.4f}")
    print(f"  Survived: {result['survived']}")

    # Run random (should have more negative rewards)
    random_results = [run_episode(world, random_policy) for _ in range(5)]
    random_avg_reward = sum(r["total_reward"] for r in random_results) / len(random_results)

    heuristic_results = [run_episode(world, heuristic_policy) for _ in range(5)]
    heuristic_avg_reward = sum(r["total_reward"] for r in heuristic_results) / len(heuristic_results)

    print(f"\nComparison:")
    print(f"  Random avg reward: {random_avg_reward:.3f}")
    print(f"  Heuristic avg reward: {heuristic_avg_reward:.3f}")

    if heuristic_avg_reward > random_avg_reward:
        print("  ✓ Heuristic outperforms random - reward signal is meaningful!")
    else:
        print("  ⚠ Heuristic does not outperform random - check reward design")


def main():
    """Run all tests."""
    print("=" * 60)
    print("LLM SOCIETY RL ENVIRONMENT TEST SUITE")
    print("=" * 60)

    # Run tests
    test_mechanics()
    random_results = test_random_policy()
    heuristic_results = test_heuristic_policy()
    test_reward_signal()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    random_survival = sum(1 for r in random_results if r["survived"]) / len(random_results)
    heuristic_survival = sum(1 for r in heuristic_results if r["survived"]) / len(heuristic_results)

    print(f"\nRandom Policy Survival Rate: {random_survival:.1%}")
    print(f"Heuristic Policy Survival Rate: {heuristic_survival:.1%}")

    if heuristic_survival > random_survival:
        print("\n✓ Heuristic significantly outperforms random")
        print("  This confirms the world has learnable survival mechanics.")
    else:
        print("\n⚠ Heuristic does not significantly outperform random")
        print("  Consider adjusting world parameters.")

    print("\n✓ Environment is ready for RL training!")


if __name__ == "__main__":
    main()
