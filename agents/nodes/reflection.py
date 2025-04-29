from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from ..core.memory import MemoryManager

class ActionOutcome(BaseModel):
    """Structured representation of an action's outcome"""
    action_type: str
    parameters: Dict
    success: bool
    reward: float
    state_changes: Dict
    timestamp: datetime = datetime.now()

class ReflectionNode:
    """Analyzes outcomes and updates agent memory"""
    
    def __init__(self, llm: ChatOpenAI, memory_manager: MemoryManager):
        self.llm = llm
        self.memory = memory_manager
        
        self.reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a reflective analyzer for a civilization game agent.
            Analyze the outcome of actions and extract strategic insights.
            Consider:
            1. Action effectiveness
            2. Strategy alignment
            3. Resource efficiency
            4. Unexpected outcomes
            5. Learning opportunities
            
            Provide structured insights that can inform future decisions."""),
            ("human", """Action taken: {action}
            Outcome: {outcome}
            Previous state: {prev_state}
            Current state: {curr_state}
            Strategic plan: {plan}
            
            Analyze the outcome and provide strategic insights:""")
        ])
        
        self.reflection_chain = LLMChain(
            llm=llm,
            prompt=self.reflection_prompt
        )
    
    async def reflect(
        self,
        action: Dict,
        outcome: Dict,
        prev_state: Dict,
        curr_state: Dict,
        plan: Dict
    ) -> None:
        """Process action outcome and update memory"""
        
        # Get reflection insights
        reflection = await self.reflection_chain.arun(
            action=str(action),
            outcome=str(outcome),
            prev_state=str(prev_state),
            curr_state=str(curr_state),
            plan=str(plan)
        )
        
        # Structure the outcome
        structured_outcome = ActionOutcome(
            action_type=action["action_type"],
            parameters=action["parameters"],
            success=outcome.get("success", False),
            reward=outcome.get("reward", 0.0),
            state_changes=self._compute_state_changes(prev_state, curr_state)
        )
        
        # Update memories
        await self._update_memories(structured_outcome, reflection)
    
    def _compute_state_changes(self, prev_state: Dict, curr_state: Dict) -> Dict:
        """Compute meaningful changes between states"""
        changes = {}
        
        # Track resource changes
        if "resources" in prev_state and "resources" in curr_state:
            changes["resource_changes"] = {
                k: curr_state["resources"].get(k, 0) - prev_state["resources"].get(k, 0)
                for k in set(curr_state["resources"]) | set(prev_state["resources"])
            }
        
        # Track unit changes
        if "units" in prev_state and "units" in curr_state:
            prev_units = len(prev_state["units"])
            curr_units = len(curr_state["units"])
            changes["unit_delta"] = curr_units - prev_units
        
        # Track city changes
        if "cities" in prev_state and "cities" in curr_state:
            prev_cities = len(prev_state["cities"])
            curr_cities = len(curr_state["cities"])
            changes["city_delta"] = curr_cities - prev_cities
        
        # Track territory changes
        if "map" in prev_state and "map" in curr_state:
            # Simplified territory comparison - actual implementation would be more sophisticated
            changes["territory_changed"] = prev_state["map"] != curr_state["map"]
        
        return changes
    
    async def _update_memories(self, outcome: ActionOutcome, reflection: str) -> None:
        """Update different memory types based on outcome"""
        
        # Create memory entry
        memory_entry = {
            "type": "action_outcome",
            "action": {
                "type": outcome.action_type,
                "parameters": outcome.parameters
            },
            "outcome": {
                "success": outcome.success,
                "reward": outcome.reward,
                "changes": outcome.state_changes
            },
            "reflection": reflection,
            "timestamp": outcome.timestamp.isoformat()
        }
        
        # Add to memory manager
        await self.memory.add_memory(
            entry_type="action_outcome",
            content=memory_entry,
            metadata={
                "success": outcome.success,
                "reward": outcome.reward,
                "action_type": outcome.action_type
            }
        )
        
    async def get_learned_insights(self) -> str:
        """Retrieve consolidated insights from memory"""
        return await self.memory.get_strategic_insights()