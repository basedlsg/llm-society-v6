"""
Memory/Cognitive Agent Mixin - Memory management and LLM reasoning
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from llm_society.utils.config import Config

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """Agent memory item"""

    content: str
    timestamp: float
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    agent_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        data["tags"] = list(data.get("tags", []))
        return cls(**data)


class MemoryMixin:
    """Mixin providing memory and cognitive capabilities to agents"""

    memories: List[Memory]
    memory_size: int
    unique_id: str
    model: Any
    config: "Config"
    llm_coordinator: Any
    llm_cache: Dict[str, str]
    conversation_history: List[Dict[str, str]]
    last_llm_call: float
    last_market_research_result: Optional[str]
    last_banking_statement: Optional[str]

    def _init_memory(self, config: "Config", llm_coordinator: Any):
        """Initialize memory attributes"""
        self.llm_coordinator = llm_coordinator
        self.memories = []
        self.memory_size = config.agents.memory_size
        self.last_llm_call = 0.0
        self.llm_cache = {}
        self.conversation_history = []
        self.last_market_research_result = None
        self.last_banking_statement = None

    async def _add_memory(
        self, content: str, importance: float = 0.5, tags: List[str] = None
    ):
        """Add a memory to the agent's memory system and save to DB."""
        memory = Memory(
            agent_id=self.unique_id,
            content=content,
            timestamp=self.model.current_step,
            importance=importance,
            tags=tags or [],
        )
        self.memories.append(memory)

        # Save to database
        if hasattr(self.model, "database_handler") and self.model.database_handler:
            try:
                await self.model.database_handler.save_agent_memory(
                    agent_id=self.unique_id,
                    memory_data=memory.to_dict(),
                    step=self.model.current_step,
                )
            except Exception as e:
                logger.error(
                    f"Agent {self.unique_id}: Failed to save memory to DB: {e}",
                    exc_info=True,
                )

        # Keep memory size manageable (in-memory list)
        if len(self.memories) > self.memory_size:
            self.memories.sort(key=lambda m: m.importance, reverse=True)
            self.memories = self.memories[: self.memory_size]

    async def _format_recent_memories(self, count: int = 3) -> str:
        """Format recent memories for LLM context"""
        recent = sorted(self.memories, key=lambda m: m.timestamp, reverse=True)[:count]
        if not recent:
            return "No recent memories."

        return "; ".join([m.content for m in recent])

    async def _manage_memory(self):
        """Clean up old or unimportant memories"""
        if len(self.memories) > self.memory_size * 1.2:
            # Aggressive cleanup
            self.memories = sorted(
                self.memories, key=lambda m: m.importance, reverse=True
            )
            self.memories = self.memories[: self.memory_size]

    async def _generate_conversation(self, other_agent: Any) -> str:
        """Generate conversation with another agent"""
        try:
            my_memories = await self._format_recent_memories()
            other_memories = await other_agent._format_recent_memories()

            prompt = f"""
You ({self.persona}) are talking with {other_agent.unique_id} ({other_agent.persona}).

Your recent memories: {my_memories}
Their recent activities: {other_memories}

Generate a brief conversation (1-2 lines) between you two.
"""

            response = await self.llm_coordinator.get_response(
                agent_id=self.unique_id, prompt=prompt, max_tokens=100
            )

            return response.strip()

        except Exception as e:
            logger.warning(f"Conversation generation failed: {e}")
            return "Had a pleasant chat about daily activities."
