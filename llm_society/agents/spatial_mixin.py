"""
Spatial Agent Mixin - Movement and positioning capabilities
"""

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from llm_society.utils.config import Config

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """3D position with utilities"""

    x: float
    y: float
    z: float = 0.0

    def distance_to(self, other: "Position") -> float:
        """Calculate Euclidean distance to another position"""
        return np.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )

    def move_towards(self, target: "Position", speed: float) -> "Position":
        """Move towards target position with given speed"""
        direction = np.array([target.x - self.x, target.y - self.y, target.z - self.z])
        distance = np.linalg.norm(direction)

        if distance <= speed:
            return Position(target.x, target.y, target.z)

        direction = direction / distance * speed
        return Position(
            self.x + direction[0], self.y + direction[1], self.z + direction[2]
        )

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        return cls(x=data.get("x", 0.0), y=data.get("y", 0.0), z=data.get("z", 0.0))


class SpatialMixin:
    """Mixin providing spatial/movement capabilities to agents"""

    position: Position
    target_position: Optional[Position]
    movement_speed: float
    config: "Config"
    energy: float

    def _init_spatial(
        self,
        config: "Config",
        position: Optional[Position] = None,
    ):
        """Initialize spatial attributes"""
        self.position = position or Position(
            random.uniform(0, config.simulation.world_size[0]),
            random.uniform(0, config.simulation.world_size[1]),
            0.0,
        )
        self.target_position = None
        self.movement_speed = config.agents.movement_speed

    async def _execute_move(self, target: Position):
        """Execute movement action"""
        self.target_position = target
        self.position = self.position.move_towards(target, self.movement_speed)
        self.energy -= 0.01  # Movement costs energy

        await self._add_memory(
            f"Moved towards ({target.x:.1f}, {target.y:.1f})", importance=0.1
        )

    def _get_nearby_agents(self):
        """Get agents within social radius"""
        nearby = []
        social_r = getattr(self.config.agents, "social_radius", 10.0)
        for agent in self.model.schedule.agents:
            if (
                agent.unique_id != self.unique_id
                and self.position.distance_to(agent.position) <= social_r
            ):
                nearby.append(agent)
        return nearby

    def _get_nearby_objects(self):
        """Get objects within interaction radius"""
        if hasattr(self.model, "get_objects_near") and self.position:
            interaction_r = getattr(self.config.agents, "interaction_radius", 5.0)
            return self.model.get_objects_near(self.position, interaction_r)
        return []
