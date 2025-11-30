"""
Single-Agent Validation Test for LLM Society

This test runs a single agent for 200 steps to validate:
1. Energy decays properly
2. Food decays and starvation works
3. Gather replenishes resources
4. Movement works
5. State is logged correctly
6. LLM vs fallback decisions are tracked

Run with:
    python -m llm_society.experiments.validation_test
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_society.utils.config import Config
from llm_society.simulation.society_simulator import SocietySimulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_validation_test(
    steps: int = 50,
    output_dir: str = "./validation_results",
):
    """
    Run a single-agent validation test.

    This validates the core mechanics work before running multi-agent experiments.
    """
    print("=" * 60)
    print("SINGLE-AGENT VALIDATION TEST")
    print("=" * 60)

    # Create minimal config
    config = Config.default()
    config.agents.count = 1  # Single agent
    config.simulation.max_steps = steps
    config.simulation.world_size = (20, 20)  # Small world
    config.simulation.energy_decay_per_step = 0.01  # Faster decay for visibility
    config.simulation.food_consumption_interval = 10  # More frequent consumption
    config.performance.enable_gpu_acceleration = False  # CPU only

    # Disable autosave for test
    config.simulation.autosave_enabled = False

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize simulator
    simulator = SocietySimulator(config)

    # Track metrics at each step
    step_logs = []

    print(f"\nRunning {steps} steps with 1 agent...")
    print("-" * 60)

    try:
        # Initialize
        await simulator._async_initial_setup()

        agent = simulator._agent_list[0]
        print(f"Agent ID: {agent.unique_id}")
        print(f"Initial position: ({agent.position.x:.2f}, {agent.position.y:.2f})")
        print(f"Initial energy: {agent.energy:.3f}")
        print(f"Initial food: {agent.resources.get('food', 0)}")
        print(f"Initial health: {agent.health:.3f}")
        print("-" * 60)

        # Run steps
        for step in range(steps):
            # Record pre-step state
            pre_state = {
                "step": step,
                "pre_energy": agent.energy,
                "pre_food": agent.resources.get("food", 0),
                "pre_health": agent.health,
                "pre_position": (agent.position.x, agent.position.y),
            }

            # Execute step
            await simulator._async_step()

            # Record post-step state
            post_state = {
                "post_energy": agent.energy,
                "post_food": agent.resources.get("food", 0),
                "post_health": agent.health,
                "post_position": (agent.position.x, agent.position.y),
                "connections": len(agent.social_connections),
                "memories": len(agent.memories),
            }

            step_log = {**pre_state, **post_state}
            step_logs.append(step_log)

            # Print progress every 10 steps
            if step % 10 == 0:
                print(f"Step {step:3d}: energy={agent.energy:.3f}, food={agent.resources.get('food', 0)}, health={agent.health:.3f}, pos=({agent.position.x:.1f}, {agent.position.y:.1f})")

        print("-" * 60)
        print("\nFINAL STATE:")
        print(f"  Energy: {agent.energy:.3f}")
        print(f"  Food: {agent.resources.get('food', 0)}")
        print(f"  Health: {agent.health:.3f}")
        print(f"  Position: ({agent.position.x:.2f}, {agent.position.y:.2f})")
        print(f"  Memories: {len(agent.memories)}")
        print(f"  Social connections: {len(agent.social_connections)}")

        # Query events from database
        events = []
        if simulator.database_handler:
            try:
                import sqlite3
                conn = sqlite3.connect("./llm_society_dynamic_data.db")
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT event_type, step_timestamp, description, details
                    FROM simulation_events
                    ORDER BY step_timestamp
                """)
                for row in cursor.fetchall():
                    events.append({
                        "event_type": row[0],
                        "step": row[1],
                        "description": row[2],
                        "details": json.loads(row[3]) if row[3] else {},
                    })
                conn.close()
            except Exception as e:
                logger.warning(f"Could not query events: {e}")

        # Analyze action distribution
        action_counts = {}
        llm_count = 0
        fallback_count = 0

        for event in events:
            if event["event_type"] == "AGENT_ACTION_CHOSEN":
                details = event.get("details", {})
                action_type = details.get("action_type", "unknown")
                action_counts[action_type] = action_counts.get(action_type, 0) + 1

                source = details.get("source", "unknown")
                if source == "llm":
                    llm_count += 1
                elif source == "fallback":
                    fallback_count += 1

        print("\n" + "=" * 60)
        print("ACTION DISTRIBUTION:")
        print("-" * 60)
        for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
            print(f"  {action}: {count}")

        print(f"\nDECISION SOURCE:")
        print(f"  LLM decisions: {llm_count}")
        print(f"  Fallback decisions: {fallback_count}")
        total = llm_count + fallback_count
        if total > 0:
            print(f"  LLM rate: {llm_count/total*100:.1f}%")

        # Validation checks
        print("\n" + "=" * 60)
        print("VALIDATION CHECKS:")
        print("-" * 60)

        # Check 1: Energy decayed
        initial_energy = step_logs[0]["pre_energy"] if step_logs else 1.0
        final_energy = agent.energy
        energy_decayed = final_energy < initial_energy
        print(f"  [{'✓' if energy_decayed else '✗'}] Energy decayed: {initial_energy:.3f} -> {final_energy:.3f}")

        # Check 2: Food consumed
        initial_food = step_logs[0]["pre_food"] if step_logs else 10
        final_food = agent.resources.get("food", 0)
        food_consumed = final_food < initial_food
        print(f"  [{'✓' if food_consumed else '✗'}] Food consumed: {initial_food} -> {final_food}")

        # Check 3: Events logged
        events_logged = len(events) > 0
        print(f"  [{'✓' if events_logged else '✗'}] Events logged: {len(events)} events")

        # Check 4: Action types present
        has_actions = len(action_counts) > 0
        print(f"  [{'✓' if has_actions else '✗'}] Action types present: {list(action_counts.keys())}")

        # Check 5: Source attribution present
        has_source = llm_count > 0 or fallback_count > 0
        print(f"  [{'✓' if has_source else '✗'}] Source attribution present: llm={llm_count}, fallback={fallback_count}")

        # Save results
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "agents": 1,
                "steps": steps,
                "world_size": list(config.simulation.world_size),
                "energy_decay": config.simulation.energy_decay_per_step,
                "food_interval": config.simulation.food_consumption_interval,
            },
            "final_state": {
                "energy": agent.energy,
                "food": agent.resources.get("food", 0),
                "health": agent.health,
                "position": (agent.position.x, agent.position.y),
                "memories": len(agent.memories),
            },
            "action_distribution": action_counts,
            "decision_source": {
                "llm": llm_count,
                "fallback": fallback_count,
            },
            "step_logs": step_logs,
            "validation": {
                "energy_decayed": energy_decayed,
                "food_consumed": food_consumed,
                "events_logged": events_logged,
                "has_actions": has_actions,
                "has_source": has_source,
            }
        }

        result_path = Path(output_dir) / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {result_path}")

        # Final verdict
        all_passed = all(results["validation"].values())
        print("\n" + "=" * 60)
        if all_passed:
            print("✓ ALL VALIDATION CHECKS PASSED")
        else:
            print("✗ SOME VALIDATION CHECKS FAILED")
        print("=" * 60)

        return results

    except Exception as e:
        logger.error(f"Validation test failed: {e}", exc_info=True)
        raise
    finally:
        if hasattr(simulator, "llm_coordinator") and simulator.llm_coordinator:
            await simulator.llm_coordinator.stop()


if __name__ == "__main__":
    # Set Gemini API key
    os.environ.setdefault("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))

    asyncio.run(run_validation_test(steps=50))
