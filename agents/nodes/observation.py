from typing import Dict, List, Optional
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
import numpy as np
import logging

# Initialize logger
logger = logging.getLogger(__name__)

class ObservationProcessor:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the current game state for a civilization-style game.
            The map is 3D one-hot encoded with:
            - First 3 values: resources (energy, materials, water)
            - Next 3*num_agents values: cities, warriors, settlers per player
            - Final num_agents values: tile ownership per player
            
            Consider:
            1. Resource availability and distribution
            2. Unit positions and opportunities
            3. City locations and development
            4. Territory control
            5. Fog of war limitations"""),
            ("human", """Current state:
            Map data: {map_data}
            Units: {units}
            Cities: {cities}
            Money: {money}
            
            Provide strategic analysis:""")
        ])
        
        self.analysis_chain = LLMChain(
            llm=llm,
            prompt=self.analysis_prompt
        )
    
    async def process(self, raw_observation: Dict) -> Dict:
        """Process raw observation into analyzed game state"""
        try:
            logger.info(f"Processing raw observation for agent: {raw_observation}")

            # Extract state components
            map_data = raw_observation.get('map', np.zeros((1,1,1)))
            units = raw_observation.get('units', [])
            cities = raw_observation.get('cities', [])
            money = raw_observation.get('money', 0)

            logger.debug(f"Extracted map_data: {map_data.shape}, units: {len(units)}, cities: {len(cities)}, money: {money}")

            # Generate analysis
            analysis = await self.analysis_chain.arun(
                map_data=self._process_map_data(map_data),
                units=self._process_units(units),
                cities=self._process_cities(cities),
                money=str(money)
            )

            logger.info(f"Generated analysis: {analysis}")

            return {
                "raw_state": raw_observation,
                "analysis": analysis,
                "statistics": {
                    "num_units": len(units) if isinstance(units, list) else units.shape[0],
                    "num_cities": len(cities) if isinstance(cities, list) else cities.shape[0],
                    "money": money,
                    "visible_resources": self._count_visible_resources(map_data)
                }
            }
        except Exception as e:
            logger.error(f"Error processing observation for agent: {raw_observation}. Error: {e}")
            return {
                "raw_state": raw_observation,
                "analysis": f"Error processing observation: {str(e)}",
                "statistics": {}
            }
    
    def _process_map_data(self, map_data: np.ndarray) -> Dict:
        """Convert map tensor into readable format"""
        try:
            logger.debug(f"Processing map data with shape: {map_data.shape}")
            resources = map_data[:,:,:3]
            processed_data = {
                "energy_resources": np.sum(resources[:,:,0]),
                "materials": np.sum(resources[:,:,1]),
                "water": np.sum(resources[:,:,2])
            }
            logger.debug(f"Processed map data: {processed_data}")
            return processed_data
        except Exception as e:
            logger.error(f"Error processing map data: {e}")
            return {"energy_resources": 0, "materials": 0, "water": 0}
    
    def _process_units(self, units) -> str:
        """Format unit information"""
        try:
            logger.debug(f"Processing units: {units}")
            if isinstance(units, list):
                return f"Total units: {len(units)}"
            return "No units available"
        except Exception as e:
            logger.error(f"Error processing units: {e}")
            return "Error processing units"
    
    def _process_cities(self, cities) -> str:
        """Format city information"""
        try:
            logger.debug(f"Processing cities: {cities}")
            if isinstance(cities, list):
                return f"Total cities: {len(cities)}"
            return "No cities available"
        except Exception as e:
            logger.error(f"Error processing cities: {e}")
            return "Error processing cities"
    
    def _count_visible_resources(self, map_data: np.ndarray) -> Dict:
        """Count visible resources in explored areas"""
        try:
            logger.debug(f"Counting visible resources in map data with shape: {map_data.shape}")
            resources = map_data[:,:,:3]
            visible_resources = {
                "energy": int(np.sum(resources[:,:,0])),
                "materials": int(np.sum(resources[:,:,1])),
                "water": int(np.sum(resources[:,:,2]))
            }
            logger.debug(f"Counted visible resources: {visible_resources}")
            return visible_resources
        except Exception as e:
            logger.error(f"Error counting visible resources: {e}")
            return {"energy": 0, "materials": 0, "water": 0}