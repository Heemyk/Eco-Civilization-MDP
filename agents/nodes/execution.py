from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from env.config.constants import ActionTypes
from env import UNIT_TYPE_MAPPING

class ActionExecutor:
    """Selects and executes actions based on strategic plan"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
        self.action_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an action selector for a civilization game agent.
            Given the current strategic plan and game state, determine the best action to take.
            
            Available actions:
            - MOVE_UNIT: Requires unit_id and direction (N,S,E,W)
            - BUILD_CITY: Requires unit_id (must be settler)
            - TRAIN_UNIT: Requires city_id and unit_type
            - START_PROJECT: Requires city_id and project_id
            
            Consider:
            1. Action validity for current state
            2. Resource requirements and availability
            3. Strategic alignment
            4. Environmental impact
            5. Long-term benefits
            
            Format response as structured action with parameters."""),
            ("human", """Current state: {state}
            Available units: {units}
            Available cities: {cities}
            Strategic plan: {plan}
            Available actions: {available_actions}
            
            Select optimal action:""")
        ])
        
    async def execute_action(
        self,
        action_type: ActionTypes,
        parameters: Dict,
        env
    ) -> Dict:
        """Execute selected action in environment"""
        try:
            # Map action to environment interface
            if action_type == ActionTypes.MOVE_UNIT:
                result = env.step({
                    "action_type": 0,  # MOVE_UNIT
                    "unit_id": parameters["unit_id"],
                    "direction": parameters["direction"],
                    "city_id": 0,  # Ignored for MOVE_UNIT
                    "project_id": 0  # Ignored for MOVE_UNIT
                })
            elif action_type == ActionTypes.BUILD_CITY:
                result = env.step({
                    "action_type": 1,  # BUILD_CITY
                    "unit_id": parameters["unit_id"],
                    "direction": "N",  # Not used for BUILD_CITY
                    "city_id": 0,
                    "project_id": 0
                })
            elif action_type == ActionTypes.TRAIN_UNIT:
                result = env.step({
                    "action_type": 2,  # TRAIN_UNIT
                    "unit_id": 0,
                    "direction": "N",
                    "city_id": parameters["city_id"],
                    "project_id": UNIT_TYPE_MAPPING[parameters["unit_type"]]
                })
            elif action_type == ActionTypes.START_PROJECT:
                result = env.step({
                    "action_type": 3,  # START_PROJECT
                    "unit_id": 0,
                    "direction": "N",
                    "city_id": parameters["city_id"],
                    "project_id": parameters["project_id"]
                })
            
            return {
                "success": True,
                "action_type": action_type,
                "parameters": parameters,
                "result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "action_type": action_type,
                "parameters": parameters,
                "error": str(e)
            }