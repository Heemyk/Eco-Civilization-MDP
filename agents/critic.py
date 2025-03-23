from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from base import *
import numpy as np

class CriticSuggestion(BaseModel):
    action: Action
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0,
        le=1
    )
    reasoning: str = Field(
        description="Detailed explanation for the suggested action"
    )
    exploration_value: float = Field(
        description="Expected exploration value (0-1)",
        ge=0,
        le=1
    )
    exploitation_value: float = Field(
        description="Expected exploitation value (0-1)",
        ge=0,
        le=1
    )

class CriticFeedback(BaseModel):
    feedback: str
    timestamp: datetime
    action_taken: Action
    outcome: str
    success: bool

class BaseCritic:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.feedback_history: List[CriticFeedback] = []
        self.parser = PydanticOutputParser(pydantic_object=CriticSuggestion)

    async def incorporate_feedback(
        self,
        feedback: str,
        previous_suggestion: Action,
        success: bool
    ):
        feedback_entry = CriticFeedback(
            feedback=feedback,
            timestamp=datetime.now(),
            action_taken=previous_suggestion,
            outcome=feedback,
            success=success
        )
        self.feedback_history.append(feedback_entry)

class ExplorationCritic(BaseCritic):
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an exploration-focused critic in a civilization-like game. 
            Your role is to suggest actions that prioritize:
            1. Exploring unknown territories
            2. Trying new strategies
            3. Taking calculated risks
            4. Discovering resources and opportunities
            5. Building in unexplored areas

            Format your response as a CriticSuggestion with high exploration_value.
            """),
            ("human", """Current game state:
            {game_state}
            
            Agent ID: {agent_id}
            
            Recent feedback history:
            {feedback_history}
            
            Suggest an action that maximizes exploration potential while remaining viable.
            {format_instructions}
            """)
        ])

    async def suggest_action(
        self,
        global_state: dict,
        agent_id: str
    ) -> CriticSuggestion:
        formatted_history = "\n".join(
            f"- {f.timestamp}: {f.feedback} (Success: {f.success})"
            for f in self.feedback_history[-5:]  # Last 5 feedback entries
        )

        response = await self.llm.agenerate([
            self.prompt_template.format_messages(
                game_state=self._format_state(global_state),
                agent_id=agent_id,
                feedback_history=formatted_history,
                format_instructions=self.parser.get_format_instructions()
            )
        ])

        return self.parser.parse(response.generations[0][0].text)

    def _format_state(self, state: dict) -> str:
        """
        Convert the game state dictionary into a human-readable string format.
        The state contains map information, units, cities, and resources.
        
        Args:
            state (dict): Contains 'map', 'units', 'cities', 'money' for all agents
        
        Returns:
            str: Formatted string representation of the state
        """
        formatted_output = []
        
        # Format map information
        if 'map' in state:
            map_data = state['map']
            height, width, channels = map_data.shape
            
            # Calculate number of agents from the map structure
            num_agents = (channels - 3) // 4  # Subtracting resource channels (3) and dividing by unit types (4)
            
            formatted_output.append("Map Information:")
            
            # Territory ownership
            ownership_info = []
            for agent_idx in range(num_agents):
                owned_tiles = np.sum(map_data[:, :, agent_idx])
                ownership_info.append(f"Agent {agent_idx}: {owned_tiles} tiles")
            formatted_output.append("Territory Control: " + ", ".join(ownership_info))
            
            # Resource information
            resource_channels_start = num_agents + (3 * num_agents)
            resources = np.sum(map_data[:, :, resource_channels_start])
            materials = np.sum(map_data[:, :, resource_channels_start + 1])
            water = np.sum(map_data[:, :, resource_channels_start + 2])
            formatted_output.append(f"Resources: {resources}, Materials: {materials}, Water Sources: {water}")

        # Format units information
        if 'units' in state:
            formatted_output.append("\nUnit Positions:")
            for agent_idx in range(num_agents):
                unit_base_idx = num_agents + (3 * agent_idx)
                warriors = np.sum(map_data[:, :, unit_base_idx + 1])
                settlers = np.sum(map_data[:, :, unit_base_idx + 2])
                formatted_output.append(f"Agent {agent_idx}: {warriors} warriors, {settlers} settlers")

        # Format cities information
        if 'cities' in state:
            formatted_output.append("\nCities:")
            for agent_idx in range(num_agents):
                city_channel = num_agents + (3 * agent_idx)
                num_cities = np.sum(map_data[:, :, city_channel])
                formatted_output.append(f"Agent {agent_idx}: {num_cities} cities")

        # Format money/economy information
        if 'money' in state:
            formatted_output.append("\nEconomy:")
            for agent_idx, money in enumerate(state['money']):
                formatted_output.append(f"Agent {agent_idx} Money: {money}")

        # Add visibility information if available
        if 'visibility_maps' in state:
            formatted_output.append("\nExploration Status:")
            for agent_idx, visibility_map in enumerate(state['visibility_maps']):
                explored_tiles = np.sum(visibility_map)
                total_tiles = visibility_map.size
                exploration_percentage = (explored_tiles / total_tiles) * 100
                formatted_output.append(f"Agent {agent_idx}: {exploration_percentage:.1f}% map explored")

        return "\n".join(formatted_output)

class ExploitationCritic(BaseCritic):
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an exploitation-focused critic in a civilization-like game. 
            Your role is to suggest actions that prioritize:
            1. Maximizing immediate rewards
            2. Protecting existing resources
            3. Optimizing current strategies
            4. Strengthening established positions
            5. Efficient resource utilization

            Format your response as a CriticSuggestion with high exploitation_value.
            """),
            ("human", """Current game state:
            {game_state}
            
            Agent ID: {agent_id}
            
            Recent success history:
            {feedback_history}
            
            Suggest an action that maximizes immediate advantages and reinforces successful strategies.
            {format_instructions}
            """)
        ])

    async def suggest_action(
        self,
        global_state: dict,
        agent_id: str
    ) -> CriticSuggestion:
        # Similar to ExplorationCritic but focuses on exploitation
        formatted_history = "\n".join(
            f"- {f.timestamp}: {f.feedback} (Success: {f.success})"
            for f in self.feedback_history[-5:] if f.success
        )

        response = await self.llm.agenerate([
            self.prompt_template.format_messages(
                game_state=self._format_state(global_state),
                agent_id=agent_id,
                feedback_history=formatted_history,
                format_instructions=self.parser.get_format_instructions()
            )
        ])

        return self.parser.parse(response.generations[0][0].text)

class AssessorCritic(BaseCritic):
    def __init__(self, llm: ChatOpenAI):
        super().__init__(llm)
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an assessor critic in a civilization-like game.
            Your role is to evaluate and reconcile suggestions from exploration and exploitation critics.
            Consider:
            1. Current game state and context
            2. Balance between exploration and exploitation
            3. Past success rates of different strategies
            4. Long-term strategic implications
            5. Risk-reward trade-offs

            Format your response as a CriticSuggestion with balanced values.
            """),
            ("human", """Current game state:
            {game_state}
            
            Explorer's suggestion:
            {explorer_suggestion}
            
            Exploiter's suggestion:
            {exploiter_suggestion}
            
            Recent history:
            {feedback_history}
            
            Evaluate both suggestions and provide a final recommendation.
            {format_instructions}
            """)
        ])

    async def reconcile_suggestions(
        self,
        explorer_suggestion: CriticSuggestion,
        exploiter_suggestion: CriticSuggestion,
        global_state: dict
    ) -> CriticSuggestion:
        formatted_history = "\n".join(
            f"- {f.timestamp}: {f.feedback} (Success: {f.success})"
            for f in self.feedback_history[-5:]
        )

        response = await self.llm.agenerate([
            self.prompt_template.format_messages(
                game_state=self._format_state(global_state),
                explorer_suggestion=explorer_suggestion.model_dump_json(),
                exploiter_suggestion=exploiter_suggestion.model_dump_json(),
                feedback_history=formatted_history,
                format_instructions=self.parser.get_format_instructions()
            )
        ])

        final_suggestion = self.parser.parse(response.generations[0][0].text)

        # Add metadata about the decision process
        final_suggestion.reasoning += f"\n\nThis decision reconciles:\n" \
                                   f"Explorer ({explorer_suggestion.confidence}): {explorer_suggestion.reasoning}\n" \
                                   f"Exploiter ({exploiter_suggestion.confidence}): {exploiter_suggestion.reasoning}"

        return final_suggestion

    def _analyze_suggestion_history(self) -> Dict[str, float]:
        """Analyze the success rate of different types of suggestions"""
        if not self.feedback_history:
            return {"exploration": 0.5, "exploitation": 0.5}

        recent_history = self.feedback_history[-20:]  # Look at last 20 decisions
        exploration_success = sum(1 for f in recent_history 
                                if f.success and f.action_taken.exploration_value > 0.6)
        exploitation_success = sum(1 for f in recent_history 
                                 if f.success and f.action_taken.exploitation_value > 0.6)
        
        total = len(recent_history) or 1  # Avoid division by zero
        return {
            "exploration": exploration_success / total,
            "exploitation": exploitation_success / total
        }

class TripletCritic:
    def __init__(
        self,
        explorer_llm: ChatOpenAI,
        exploiter_llm: ChatOpenAI,
        assessor_llm: ChatOpenAI
    ):
        self.explorer = ExplorationCritic(explorer_llm)
        self.exploiter = ExploitationCritic(exploiter_llm)
        self.assessor = AssessorCritic(assessor_llm)
        
    async def generate_suggestion(
        self, 
        global_state: dict,
        agent_id: str
    ) -> Action:
        # Get suggestions from both critics
        explorer_suggestion = await self.explorer.suggest_action(global_state, agent_id)
        exploiter_suggestion = await self.exploiter.suggest_action(global_state, agent_id)
        
        # Let assessor make final decision
        final_suggestion = await self.assessor.reconcile_suggestions(
            explorer_suggestion,
            exploiter_suggestion,
            global_state
        )
        
        return final_suggestion

    async def process_feedback(
        self, 
        feedback: str, 
        previous_suggestion: Action,
        global_state: dict
    ) -> Action:
        # Update critics with feedback
        await self.explorer.incorporate_feedback(feedback, previous_suggestion)
        await self.exploiter.incorporate_feedback(feedback, previous_suggestion)
        
        # Generate new suggestion
        return await self.generate_suggestion(global_state)
