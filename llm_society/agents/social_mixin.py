"""
Social Agent Mixin - Social interactions and relationships
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from llm_society.social.family_system import RelationshipType, Family, FamilyType

logger = logging.getLogger(__name__)


class SocialMixin:
    """Mixin providing social interaction capabilities to agents"""

    social_connections: Dict[str, float]
    social_reputation: float
    social_interactions: int
    happiness: float
    energy: float
    family_id: Optional[int]
    cultural_group_id: Optional[int]
    cultural_affinities: Dict[str, float]
    unique_id: str
    model: Any
    config: Any
    position: Any

    def _init_social(self):
        """Initialize social attributes"""
        self.social_connections = {}
        self.social_reputation = 0.5
        self.social_interactions = 0
        self.family_id = None
        self.cultural_group_id = None
        self.cultural_affinities = {
            "harmonists": 0.2,
            "builders": 0.2,
            "guardians": 0.2,
            "scholars": 0.2,
            "wanderers": 0.2,
        }

    async def _execute_talk(self, target_id: str):
        """Execute social interaction"""
        target_agent = next(
            (a for a in self.model.schedule.agents if a.unique_id == target_id), None
        )

        if target_agent:
            distance = self.position.distance_to(target_agent.position)
            if distance <= getattr(self.config.agents, "social_radius", 10.0):
                # Successful interaction
                self.social_interactions += 1
                self.happiness += 0.05

                # Generate conversation
                conversation = await self._generate_conversation(target_agent)
                await self._add_memory(
                    f"Talked with {target_id}: {conversation[:50]}...", importance=0.6
                )

                # Update relationship
                if target_id in self.social_connections:
                    self.social_connections[target_id] += 0.1
                else:
                    self.social_connections[target_id] = 0.2

    async def _execute_family_interaction_spouse(self, description: str):
        """Executes a family interaction with the agent's spouse."""
        from llm_society.social.family_system import RelationshipType

        family_member_data = self.model.family_system.family_members.get(self.unique_id)
        if not family_member_data:
            await self._add_memory(
                f"Tried spouse interaction: {description} (not in family)",
                importance=0.4,
            )
            return

        spouse_id = None
        for related_agent_id, relationship_type in family_member_data.relationships.items():
            if relationship_type == RelationshipType.SPOUSE:
                spouse_id = related_agent_id
                break

        if not spouse_id:
            await self._add_memory(
                f"Tried spouse interaction: {description} (not married)", importance=0.4
            )
            logger.info(
                f"Agent {self.unique_id} attempted to interact with spouse but is not married."
            )
            return

        family_id_str = (
            family_member_data.married_family_id or family_member_data.birth_family_id
        )
        if not family_id_str:
            await self._add_memory(
                f"Tried spouse interaction with {spouse_id}: {description} (no family ID)",
                importance=0.4,
            )
            return

        logger.info(
            f"Agent {self.unique_id} performing interaction with spouse {spouse_id} (Family {family_id_str}): {description}"
        )
        self.model.family_system.process_interaction_with_spouse(
            agent_id=self.unique_id,
            spouse_id=spouse_id,
            family_id=family_id_str,
            description=description,
            current_step=self.model.current_step,
        )
        await self._add_memory(
            f"Interacted with spouse {spouse_id}: {description}", importance=0.7
        )
        self.energy -= 0.05
        self.happiness += 0.07

    async def _execute_family_interaction_child(self, child_id: str, description: str):
        """Executes a family interaction with a specific child."""
        from llm_society.social.family_system import RelationshipType, Family, FamilyType

        family_member_data = self.model.family_system.family_members.get(self.unique_id)
        if not family_member_data:
            await self._add_memory(
                f"Tried child interaction {child_id}: {description} (not in family)",
                importance=0.4,
            )
            return

        is_child = (
            child_id in family_member_data.relationships
            and family_member_data.relationships[child_id] == RelationshipType.CHILD
        )

        if not is_child:
            await self._add_memory(
                f"Tried child interaction with {child_id} (not my child): {description}",
                importance=0.4,
            )
            logger.info(
                f"Agent {self.unique_id} attempted to interact with {child_id}, but they are not listed as a child."
            )
            return

        family_id_str = family_member_data.birth_family_id
        if (
            family_member_data.married_family_id
            and child_id
            in self.model.family_system.families.get(
                family_member_data.married_family_id,
                Family(family_id="", family_name="", family_type=FamilyType.NUCLEAR),
            ).members
        ):
            family_id_str = family_member_data.married_family_id

        if not family_id_str:
            await self._add_memory(
                f"Tried child interaction with {child_id}: {description} (no family ID context)",
                importance=0.4,
            )
            return

        logger.info(
            f"Agent {self.unique_id} performing interaction with child {child_id} (Family {family_id_str}): {description}"
        )
        self.model.family_system.process_interaction_with_child(
            agent_id=self.unique_id,
            child_id=child_id,
            family_id=family_id_str,
            description=description,
            current_step=self.model.current_step,
        )
        await self._add_memory(
            f"Interacted with child {child_id}: {description}", importance=0.65
        )
        self.energy -= 0.04
        self.happiness += 0.06

    async def _execute_family_interaction_household(self, description: str):
        """Executes a general household management task for the family."""
        family_member_data = self.model.family_system.family_members.get(self.unique_id)
        if not family_member_data:
            await self._add_memory(
                f"Tried household task: {description} (not in family)", importance=0.4
            )
            return

        family_id_str = (
            family_member_data.married_family_id or family_member_data.birth_family_id
        )
        if not family_id_str:
            await self._add_memory(
                f"Tried household task: {description} (no family ID)", importance=0.4
            )
            return

        logger.info(
            f"Agent {self.unique_id} performing household management task for family {family_id_str}: {description}"
        )
        self.model.family_system.process_household_management_task(
            agent_id=self.unique_id,
            family_id=family_id_str,
            description=description,
            current_step=self.model.current_step,
        )
        await self._add_memory(
            f"Managed household task for family {family_id_str}: {description}",
            importance=0.5,
        )
        self.energy -= 0.06
