from langchain.chat_models import ChatOpenAI
from langchain.memory import MongoDBChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from typing import List

from base import *

class LLMActor:
    def __init__(
        self, 
        agent_id: str,
        llm: ChatOpenAI,
        mongodb_connection: str,
        memory_collection: str
    ):
        self.agent_id = agent_id
        self.llm = llm
        self.short_term_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.long_term_memory = MongoDBChatMessageHistory(
            connection_string=mongodb_connection,
            session_id=agent_id,
            collection_name=memory_collection
        )
        
        self.action_template = ChatPromptTemplate.from_messages([
            ("system", "You are an agent in a civilization-like game. You need to evaluate the suggested action and either agree or disagree based on your memory and current observation."),
            ("human", "{observation}\n\nSuggested action: {suggested_action}\n\nBased on your memory and the current state, do you agree with this action? Respond with 'AGREE' or 'DISAGREE' and provide reasoning.")
        ])

    def evaluate_suggestion(
        self, 
        observation: Observation, 
        suggested_action: Action
    ) -> tuple[bool, str]:
        # Format observation and action for LLM
        formatted_obs = self._format_observation(observation)
        
        # Get response from LLM
        response = self.llm(
            self.action_template.format_messages(
                observation=formatted_obs,
                suggested_action=suggested_action.model_dump_json()
            )
        )
        
        # Parse response
        decision = "AGREE" in response.content.upper()
        reasoning = response.content
        
        # Store in memory
        self._update_memory(observation, suggested_action, reasoning)
        
        return decision, reasoning

    def _format_observation(self, obs: Observation) -> str:
        # Convert observation to text format
        return f"""Current game state:
        Visible map: {obs.map_state}
        Units: {obs.visible_units}
        Cities: {obs.visible_cities}
        Resources: {obs.resources}
        Money: {obs.current_money}"""

    def _update_memory(
        self, 
        observation: Observation, 
        action: Action, 
        reasoning: str
    ):
        # Update short-term memory
        self.short_term_memory.save_context(
            {"input": observation.model_dump_json()},
            {"output": f"Action: {action.model_dump_json()}\nReasoning: {reasoning}"}
        )
        
        # Update long-term memory periodically
        self.long_term_memory.add_message({
            "observation": observation.model_dump_json(),
            "action": action.model_dump_json(),
            "reasoning": reasoning
        })
