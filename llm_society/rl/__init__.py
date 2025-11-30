"""
LLM Society RL Module

This module provides Atropos-compatible environment wrappers for RL training
on the LLM Society survival simulation.

Key components:
- SurvivalWorld: Lightweight world for fast RL rollouts
- LlmSocietyEnv: Atropos BaseEnv implementation (requires atroposlib)
- LlmSocietyEnvConfig: Configuration for the RL environment

Usage:
    # As Atropos service
    python -m llm_society.rl.atropos_env serve --config configs/llm_society.yaml

    # Standalone test
    python -m llm_society.rl.test_env
"""

# Import standalone components that don't require atroposlib
try:
    from llm_society.rl.atropos_env import (
        SurvivalWorld,
        SurvivalWorldConfig,
    )

    # These require atroposlib, may fail
    try:
        from llm_society.rl.atropos_env import (
            LlmSocietyEnv,
            LlmSocietyEnvConfig,
        )
    except ImportError:
        LlmSocietyEnv = None
        LlmSocietyEnvConfig = None

except ImportError as e:
    # If even basic imports fail, provide fallback
    SurvivalWorld = None
    SurvivalWorldConfig = None
    LlmSocietyEnv = None
    LlmSocietyEnvConfig = None

__all__ = [
    "LlmSocietyEnv",
    "LlmSocietyEnvConfig",
    "SurvivalWorld",
    "SurvivalWorldConfig",
]
