from typing import Dict, Annotated, TypeVar
from langchain_core.messages import BaseMessage
from langgraph.graph import Graph
# Removed unused import for ToolExecutor
# from langgraph.prebuilt import ToolInvocation

from .agent import GameAgent

AgentState = TypeVar("AgentState", bound=Dict)

def create_agent_workflow(agent: GameAgent, env) -> Graph:
    """Create a workflow graph for a game agent"""
    
    def observation_node(state: AgentState) -> AgentState:
        """Process current game state observation"""
        observation = env.observe(state["agent_id"])
        processed_obs = agent.observe(observation)
        state["observation"] = processed_obs
        return state

    def planning_node(state: AgentState) -> AgentState:
        """Strategic planning based on observation"""
        plan = agent.plan(state["observation"])
        state["plan"] = plan
        return state

    def decision_node(state: AgentState) -> AgentState:
        """Tactical decision making"""
        action = agent.decide(state["observation"], state["plan"])
        state["action"] = action
        return state

    def execution_node(state: AgentState) -> AgentState:
        """Execute action in environment"""
        env.step(state["action"])
        reward = env.rewards[state["agent_id"]]
        state["reward"] = reward
        return state

    def reflection_node(state: AgentState) -> AgentState:
        """Learn from action outcomes"""
        agent.reflect(
            state["observation"], 
            state["action"],
            state["reward"]
        )
        return state

    # Create workflow graph
    workflow = Graph()

    # Add nodes
    workflow.add_node("observe", observation_node)
    workflow.add_node("plan", planning_node)
    workflow.add_node("decide", decision_node)
    workflow.add_node("execute", execution_node)
    workflow.add_node("reflect", reflection_node)

    # Define edges
    workflow.add_edge("observe", "plan")
    workflow.add_edge("plan", "decide")
    workflow.add_edge("decide", "execute")
    workflow.add_edge("execute", "reflect")
    workflow.add_edge("reflect", "observe")

    # Set entry point
    workflow.set_entry_point("observe")

    return workflow.compile()