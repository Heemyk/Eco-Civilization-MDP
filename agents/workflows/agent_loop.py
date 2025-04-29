from typing import Dict, List, Optional
# from langchain.chat_models import ChatOpenAI
from langgraph.graph import Graph
import logging

from ..core.agent import GameAgent
from ..nodes.observation import ObservationProcessor
from ..nodes.planning import StrategicPlanner
from ..nodes.execution import ActionExecutor
from ..nodes.reflection import ReflectionNode

# Initialize logger
logger = logging.getLogger(__name__)

class AgentWorkflow:
    """Main workflow implementation for game agent"""
    
    def __init__(
        self,
        agent: GameAgent,
        env,
        observation_processor: ObservationProcessor,
        planner: StrategicPlanner,
        executor: ActionExecutor,
        reflection_node: ReflectionNode
    ):
        self.agent = agent
        self.env = env
        self.observation_processor = observation_processor
        self.planner = planner
        self.executor = executor
        self.reflection_node = reflection_node
        
    async def create_workflow(self) -> Graph:
        """Create the main agent interaction workflow"""
        
        # Define workflow nodes
        async def process_observation(state: Dict) -> Dict:
            try:
                logger.info(f"Processing observation for agent {state['agent_id']}")
                raw_obs = self.env.observe(state["agent_id"])
                processed_obs = await self.observation_processor.process(raw_obs)
                state["observation"] = processed_obs
                logger.info(f"Observation processed: {processed_obs}")
                return state
            except Exception as e:
                logger.error(f"Error in process_observation: {e}")
                raise
        
        async def generate_plan(state: Dict) -> Dict:
            try:
                logger.info(f"Generating plan for agent {state['agent_id']}")
                insights = await self.agent.strategic_memory.get_messages()
                insights_text = "\n".join(str(m.content) for m in insights[-5:])
                
                # Create plan
                plan = await self.planner.create_plan(
                    state_analysis=state["observation"]["analysis"],
                    historical_insights=insights_text,
                    available_resources=state["observation"]["statistics"]
                )
                state["plan"] = plan
                logger.info(f"Plan generated: {plan}")
                return state
            except Exception as e:
                logger.error(f"Error in generate_plan: {e}")
                raise
        
        async def select_action(state: Dict) -> Dict:
            try:
                logger.info(f"Selecting action for agent {state['agent_id']}")
                action_type, parameters = await self.executor.select_action(
                    strategic_plan=state["plan"],
                    current_state=state["observation"],
                    available_actions=self.env.available_actions(state["agent_id"])
                )
                state["action"] = {
                    "type": action_type,
                    "parameters": parameters
                }
                logger.info(f"Action selected: {state['action']}")
                return state
            except Exception as e:
                logger.error(f"Error in select_action: {e}")
                raise
        
        async def execute_action(state: Dict) -> Dict:
            try:
                logger.info(f"Executing action for agent {state['agent_id']}")
                result = await self.executor.execute_action(
                    action_type=state["action"]["type"],
                    parameters=state["action"]["parameters"],
                    env=self.env
                )
                state["action_result"] = result
                logger.info(f"Action executed: {result}")
                return state
            except Exception as e:
                logger.error(f"Error in execute_action: {e}")
                raise
        
        async def process_reflection(state: Dict) -> Dict:
            try:
                logger.info(f"Processing reflection for agent {state['agent_id']}")
                await self.reflection_node.reflect(
                    action=state["action"],
                    outcome=state["action_result"],
                    prev_state=state["observation"]["raw_state"],
                    curr_state=self.env.observe(state["agent_id"]),
                    plan=state["plan"]
                )
                logger.info(f"Reflection processed for agent {state['agent_id']}")
                return state
            except Exception as e:
                logger.error(f"Error in process_reflection: {e}")
                raise
        
        # Create workflow graph
        workflow = Graph()
        
        # Add nodes
        workflow.add_node("observe", process_observation)
        workflow.add_node("plan", generate_plan)
        workflow.add_node("decide", select_action)
        workflow.add_node("execute", execute_action)
        workflow.add_node("reflect", process_reflection)
        
        # Define edges
        workflow.add_edge("observe", "plan")
        workflow.add_edge("plan", "decide")
        workflow.add_edge("decide", "execute")
        workflow.add_edge("execute", "reflect")
        workflow.add_edge("reflect", "observe")
        
        # Set entry point
        workflow.set_entry_point("observe")
        
        return workflow.compile()
    
    async def run_episode(self, max_steps: int = 100) -> Dict:
        """Run a complete game episode"""
        workflow = await self.create_workflow()
        
        state = {
            "agent_id": self.agent.agent_id,
            "step": 0,
            "total_reward": 0.0
        }
        
        try:
            while state["step"] < max_steps and not self.env.episode_done():
                # Run one step of the workflow using ainvoke
                next_state = await workflow.ainvoke(state)
                
                # Update state
                state.update(next_state)
                state["step"] += 1
                
                if "action_result" in state:
                    reward = state["action_result"].get("reward", 0.0)
                    state["total_reward"] += reward
            
            return {
                "steps": state["step"],
                "total_reward": state["total_reward"],
                "final_state": self.env.observe(state["agent_id"]),
                "action": state.get("action", None)
            }
            
        except Exception as e:
            print(f"Error running workflow: {str(e)}")
            return {
                "steps": state["step"],
                "total_reward": state["total_reward"],
                "error": str(e)
            }