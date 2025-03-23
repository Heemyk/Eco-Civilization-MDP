import os
import asyncio
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import sys
from typing import Dict
import logging
from datetime import datetime

# Add parent directory to path for importing civ
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
env_dir = os.path.join(parent_dir, 'env')
sys.path.append(env_dir)

from env.civ import Civilization
from actor import LLMActor
from critic import TripletCritic
from orchestration import GameOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'game_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_environment(map_size: tuple[int, int], num_agents: int) -> Civilization:
    """Initialize the Civilization environment."""
    try:
        env = Civilization(
            map_size=map_size,
            num_agents=num_agents,
            render_mode='human'  # Set to None for no visualization
        )
        logger.info(f"Environment initialized with map size {map_size} and {num_agents} agents")
        return env
    except Exception as e:
        logger.error(f"Failed to initialize environment: {str(e)}")
        raise

def setup_llms(temperature: float = 0.7) -> Dict[str, ChatOpenAI]:
    """Initialize different LLMs for critics and actors."""
    try:
        # Load environment variables
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Create LLMs with different temperatures for different roles
        llms = {
            "explorer": ChatOpenAI(
                temperature=temperature + 0.2,  # More creative
                model_name="gpt-4",
                api_key=api_key
            ),
            "exploiter": ChatOpenAI(
                temperature=temperature - 0.2,  # More conservative
                model_name="gpt-4",
                api_key=api_key
            ),
            "assessor": ChatOpenAI(
                temperature=temperature,  # Balanced
                model_name="gpt-4",
                api_key=api_key
            ),
            "actor": ChatOpenAI(
                temperature=temperature,
                model_name="gpt-4",
                api_key=api_key
            )
        }
        logger.info("LLMs initialized successfully")
        return llms
    except Exception as e:
        logger.error(f"Failed to initialize LLMs: {str(e)}")
        raise

def setup_actors(
    env: Civilization,
    actor_llm: ChatOpenAI,
    mongodb_uri: str
) -> Dict[str, LLMActor]:
    """Initialize actors for each agent in the environment."""
    try:
        actors = {}
        for agent_id in env.agents:
            actors[agent_id] = LLMActor(
                agent_id=str(agent_id),
                llm=actor_llm,
                mongodb_connection=mongodb_uri,
                memory_collection=f"agent_{agent_id}_memory"
            )
        logger.info(f"Initialized {len(actors)} actors")
        return actors
    except Exception as e:
        logger.error(f"Failed to initialize actors: {str(e)}")
        raise

async def main():
    try:
        # Configuration
        MAP_SIZE = (15, 30)
        NUM_AGENTS = 4
        MAX_ITERATIONS = 1000
        MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")

        # Setup components
        env = setup_environment(MAP_SIZE, NUM_AGENTS)
        llms = setup_llms()
        
        # Initialize critics
        critic = TripletCritic(
            explorer_llm=llms["explorer"],
            exploiter_llm=llms["exploiter"],
            assessor_llm=llms["assessor"]
        )
        
        # Initialize actors
        actors = setup_actors(env, llms["actor"], MONGODB_URI)
        
        # Initialize orchestrator
        orchestrator = GameOrchestrator(
            env=env,
            actors=actors,
            critic=critic,
            max_iterations=MAX_ITERATIONS
        )

        # Run episode
        logger.info("Starting game episode")
        await orchestrator.run_episode()
        logger.info("Game episode completed")

    except Exception as e:
        logger.error(f"Game failed with error: {str(e)}")
        raise
    finally:
        # Cleanup
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    try:
        # Check for required environment variables
        required_vars = ["OPENAI_API_KEY", "MONGODB_URI"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Game terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)
