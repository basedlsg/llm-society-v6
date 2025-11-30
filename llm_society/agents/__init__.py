"""
LLM Society Agents Module

This module contains the LLM-powered agent implementations and related mixins.
"""

from llm_society.agents.llm_agent import LLMAgent, AgentState, Memory, Position

# Export mixins for modular agent composition
from llm_society.agents.spatial_mixin import SpatialMixin, Position as SpatialPosition
from llm_society.agents.memory_mixin import MemoryMixin, Memory as MemoryItem
from llm_society.agents.social_mixin import SocialMixin
from llm_society.agents.economic_mixin import EconomicMixin

__all__ = [
    "LLMAgent",
    "AgentState",
    "Memory",
    "Position",
    "SpatialMixin",
    "MemoryMixin",
    "SocialMixin",
    "EconomicMixin",
]
