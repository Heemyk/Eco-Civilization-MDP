import os
import asyncio
from typing import Dict
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # Updated import
import wandb
import sys

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import from env package
from env import Civilization, ActionTypes
from env.config import UNIT_TYPE_MAPPING, MAX_AGENTS
from env.entities import Unit, City

# Import agent components
from agents.core import GameAgent, AgentConfig
from agents.core.memory import MemoryManager
from agents.nodes import (
    ObservationProcessor,
    StrategicPlanner,
    ActionExecutor,
    ReflectionNode
)
from agents.workflows import AgentWorkflow, OfflineProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all messages
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler('debug_log.log')  # Save to a file for later analysis
    ]
)

# Adjust logging level for pymongo to reduce noise
logging.getLogger('pymongo').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def setup_environment(map_size: tuple[int, int], num_agents: int) -> Civilization:
    """Initialize the Civilization environment"""
    try:
        if num_agents > MAX_AGENTS:
            raise ValueError(f"Number of agents cannot exceed {MAX_AGENTS}")
            
        env = Civilization(map_size=map_size, num_agents=num_agents)
        logger.info(f"Environment initialized with map size {map_size} and {num_agents} agents")
        return env
    except Exception as e:
        logger.error(f"Failed to initialize environment: {str(e)}")
        raise

def setup_agents(
    num_agents: int,
    mongodb_uri: str,
    openai_api_key: str
) -> Dict[str, Dict]:
    """Initialize game agents"""
    agents = {}
    
    for i in range(num_agents):
        agent_id = f"agent_{i}"
        
        # Create LLM
        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.7,
            openai_api_key=openai_api_key
        )
        
        # Create agent config
        config = AgentConfig(
            agent_id=agent_id,
            llm=llm,
            mongodb_uri=mongodb_uri
        )
        
        # Create agent components
        memory_manager = MemoryManager(llm, mongodb_uri, agent_id)
        observation_processor = ObservationProcessor(llm)
        planner = StrategicPlanner(llm)
        executor = ActionExecutor(llm)
        reflection_node = ReflectionNode(llm, memory_manager)
        
        # Create agent
        agent = GameAgent(config)
        
        # Store agent with its components
        agents[agent_id] = {
            "agent": agent,
            "components": {
                "memory": memory_manager,
                "observation": observation_processor,
                "planner": planner,
                "executor": executor,
                "reflection": reflection_node
            }
        }
    
    return agents

async def run_game_episode(
    env: Civilization,
    agents: Dict[str, Dict],
    max_steps: int = 1000
) -> Dict:
    """Run a complete game episode"""
    
    # Initialize episode metrics
    episode_metrics = {
        "steps": 0,
        "rewards": {},
        "agent_actions": {}
    }
    
    # Create agent workflows
    workflows = {}
    for agent_id, agent_data in agents.items():
        workflow = AgentWorkflow(
            agent=agent_data["agent"],
            env=env,
            observation_processor=agent_data["components"]["observation"],
            planner=agent_data["components"]["planner"],
            executor=agent_data["components"]["executor"],
            reflection_node=agent_data["components"]["reflection"]
        )
        workflows[agent_id] = workflow
    
    # Reset environment
    env.reset()
    
    # Create offline processors
    offline_processors = {
        agent_id: OfflineProcessor(
            llm=agent_data["agent"].llm,
            memory_manager=agent_data["components"]["memory"]
        )
        for agent_id, agent_data in agents.items()
    }
    
    # Main game loop
    step = 0
    while step < max_steps and not env.episode_done():
        # Run one step for each agent
        for agent_id, workflow in workflows.items():
            # Run agent workflow
            result = await workflow.run_episode(max_steps=1)
            
            # Update metrics
            episode_metrics["rewards"][agent_id] = episode_metrics["rewards"].get(agent_id, 0) + result["total_reward"]
            if agent_id not in episode_metrics["agent_actions"]:
                episode_metrics["agent_actions"][agent_id] = []
            if "action" in result:
                episode_metrics["agent_actions"][agent_id].append(result["action"])
        
        # Periodically run offline processing
        if step % 100 == 0:
            for agent_id, processor in offline_processors.items():
                offline_workflow = await processor.create_offline_workflow()
                await offline_workflow.ainvoke({})
        
        step += 1
    
    episode_metrics["steps"] = step
    return episode_metrics

async def main():
    try:
        # Configuration
        MAP_SIZE = (15, 30)
        NUM_AGENTS = 4
        MAX_EPISODES = 100
        MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Initialize WandB
        wandb.init(
            project="eco-civilization",
            config={
                "map_size": MAP_SIZE,
                "num_agents": NUM_AGENTS,
                "max_episodes": MAX_EPISODES
            }
        )
        
        # Setup components
        env = setup_environment(MAP_SIZE, NUM_AGENTS)
        agents = setup_agents(NUM_AGENTS, MONGODB_URI, OPENAI_API_KEY)
        
        # Run episodes
        for episode in range(MAX_EPISODES):
            logger.info(f"Starting episode {episode + 1}/{MAX_EPISODES}")
            
            # Run episode
            metrics = await run_game_episode(env, agents, max_steps=1000)
            
            # Log metrics
            wandb.log({
                "episode": episode,
                "total_steps": metrics["steps"],
                **{f"agent_{i}_reward": reward for i, reward in metrics["rewards"].items()}
            })
            
            logger.info(f"Episode {episode + 1} completed in {metrics['steps']} steps")
            logger.info(f"Rewards: {metrics['rewards']}")
        
        # Cleanup
        wandb.finish()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
