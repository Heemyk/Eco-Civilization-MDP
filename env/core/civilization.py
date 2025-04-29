from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector
from gymnasium import spaces

from ..config.constants import (
    ActionTypes, UNIT_TYPE_MAPPING, MAX_AGENTS, 
    WARRIOR_COST, SETTLER_COST, REWARD_WEIGHTS,
    DEFAULT_PROJECTS
)
from ..entities.unit import Unit
from ..entities.city import City
from ..utils.map_utils import (
    get_adjacent_tiles, calculate_visibility, place_resources,
    calculate_entropy, is_tile_empty, get_tile_info
)
from ..rendering.renderer import BaseRenderer, PygameRenderer, WebRenderer

class Civilization(AECEnv):
    """Core game environment for the Civilization game."""
    
    metadata = {'render_modes': ['human', 'web'], 'name': 'Civilization_v0'}

    def __init__(
        self,
        map_size: Tuple[int, int],
        num_agents: int,
        max_cities: int = 10,
        max_projects: int = 5,
        max_units_per_agent: int = 50,
        visibility_range: int = 1,
        render_mode: str = 'human'
    ):
        """Initialize the Civilization environment."""
        super().__init__()
        
        if num_agents > MAX_AGENTS:
            raise ValueError(f"Number of players ({num_agents}) exceeds maximum allowed ({MAX_AGENTS})")

        self.map_height, self.map_width = map_size
        self.num_of_agents = num_agents
        self.max_units_per_agent = max_units_per_agent
        self.max_projects = max_projects
        self.max_cities = max_cities
        self.visibility_range = visibility_range
        self.render_mode = render_mode

        # Initialize agents
        self.agents = [i for i in range(num_agents)]
        self.possible_agents = self.agents[:]
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # Initialize game state
        self._initialize_game_state()
        self._initialize_spaces()
        self._initialize_map()
        self._initialize_projects()

        # Initialize renderer
        self.renderer = (
            WebRenderer() if render_mode == 'web'
            else PygameRenderer(self.map_width, self.map_height)
        )

    def _initialize_game_state(self) -> None:
        """Initialize all game state variables."""
        self.visibility_maps = {
            agent: np.zeros((self.map_height, self.map_width), dtype=bool)
            for agent in self.agents
        }
        self.units = {agent: [] for agent in self.agents}
        self.cities = {agent: [] for agent in self.agents}
        self.money = {agent: 0 for agent in self.agents}
        self.gdp_bonus = {agent: 0 for agent in self.agents}
        self.environmental_impact = {agent: 0 for agent in self.agents}
        self.state_visit_count = {agent: {} for agent in self.agents}
        self.previous_states = {agent: None for agent in self.agents}
        
        # Combat and reward tracking
        self.last_attacker = None
        self.last_target_destroyed = False
        self.units_lost = {agent: 0 for agent in self.agents}
        self.units_eliminated = {agent: 0 for agent in self.agents}
        self.cities_lost = {agent: 0 for agent in self.agents}
        self.cities_captured = {agent: 0 for agent in self.agents}
        self.resources_gained = {agent: 0 for agent in self.agents}

    def _initialize_spaces(self) -> None:
        """Initialize observation and action spaces."""
        # Observation space
        self.observation_spaces = {
            agent: spaces.Dict({
                "map": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.map_height, self.map_width, self._calculate_num_channels()),
                    dtype=np.float32
                ),
                "units": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(self.max_units_per_agent, 4),  # x, y, health, type
                    dtype=np.float32
                ),
                "cities": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(self.max_cities, self._calculate_city_attributes()),
                    dtype=np.float32
                ),
                "money": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32
                )
            }) for agent in self.agents
        }

        # Action space
        self.action_spaces = {
            agent: spaces.Dict({
                "action_type": spaces.Discrete(7),
                "unit_id": spaces.Discrete(self.max_units_per_agent),
                "direction": spaces.Discrete(4),
                "city_id": spaces.Discrete(self.max_cities),
                "project_id": spaces.Discrete(self.max_projects)
            }) for agent in self.agents
        }

    def _initialize_map(self) -> None:
        """Initialize the game map."""
        num_channels = self._calculate_num_channels()
        self.map = np.zeros(
            (self.map_height, self.map_width, num_channels),
            dtype=np.float32
        )
        self.map = place_resources(self.map, self.num_of_agents)
        self._place_starting_units()

    def _initialize_projects(self) -> None:
        """Initialize available projects."""
        self.projects = DEFAULT_PROJECTS.copy()

    def _calculate_num_channels(self) -> int:
        """Calculate number of channels needed for the map."""
        ownership_channels = self.num_of_agents
        units_channels = 3 * self.num_of_agents  # city, warrior, settler per agent
        resource_channels = 3  # resource, material, water
        return ownership_channels + units_channels + resource_channels

    def _calculate_city_attributes(self) -> int:
        """Calculate number of attributes for city observations."""
        return (
            1 +  # health
            2 +  # x, y coordinates
            3 +  # resources (resource, material, water)
            self.max_projects +  # completed projects
            1 +  # current project
            1    # project duration
        )

    def _place_starting_units(self) -> None:
        """Place initial units for each agent."""
        for agent_idx in range(self.num_of_agents):
            placed = False
            while not placed:
                x = np.random.randint(0, self.map_width)
                y = np.random.randint(0, self.map_height)
                if is_tile_empty(self.map, x, y, self.num_of_agents):
                    self._place_unit(agent_idx, 'settler', x, y)
                    
                    # Try to place warrior adjacent to settler
                    adjacent_tiles = get_adjacent_tiles(x, y, self.map_width, self.map_height)
                    for adj_x, adj_y in adjacent_tiles:
                        if is_tile_empty(self.map, adj_x, adj_y, self.num_of_agents):
                            self._place_unit(agent_idx, 'warrior', adj_x, adj_y)
                            placed = True
                            break

    def _place_unit(self, agent_idx: int, unit_type: str, x: int, y: int) -> None:
        """Place a unit on the map."""
        unit_types = {'city': 0, 'warrior': 1, 'settler': 2}
        unit_channel = self.num_of_agents + (3 * agent_idx) + unit_types[unit_type]
        self.map[y, x, unit_channel] = 1
        
        unit = Unit(x, y, unit_type, self.agents[agent_idx], self)
        self.units[self.agents[agent_idx]].append(unit)
        self._update_visibility(self.agents[agent_idx], x, y)

    def _update_visibility(self, agent: int, x: int, y: int) -> None:
        """Update visibility map for an agent."""
        visibility = calculate_visibility(
            x, y, self.visibility_range,
            self.map_width, self.map_height
        )
        self.visibility_maps[agent] |= visibility

    def episode_done(self) -> bool:
        """Check if the episode is complete.
        
        Returns:
            bool: True if episode is done, False otherwise.
            
        Episode is done when:
        1. Only one or zero agents remain active (have units or cities)
        2. All agents are done (checked via self.dones)
        3. Maximum number of steps reached (if specified)
        """
        # Check for active agents
        active_agents = [
            agent for agent in self.agents
            if self.units[agent] or self.cities[agent]
        ]
        
        # Check if all agents are done
        all_done = all(
            self.dones.get(agent, True)
            for agent in self.possible_agents
        )
        
        # Episode is done if:
        # - 1 or fewer active agents remain (someone won or everyone lost)
        # - All agents are marked as done
        return len(active_agents) <= 1 or all_done

    def step(self, action: Dict) -> None:
        """Execute one step in the environment."""
        agent = self.agent_selection
        prev_state = self._get_state_snapshot(agent)
        
        # Process ongoing projects
        self._process_city_projects(agent)
        
        # Handle action
        action_type = action['action_type']
        if action_type == ActionTypes.MOVE_UNIT:
            self._handle_move_unit(agent, action)
        elif action_type == ActionTypes.ATTACK_UNIT:
            self._handle_attack_unit(agent, action)
        elif action_type == ActionTypes.FOUND_CITY:
            self._handle_found_city(agent, action)
        elif action_type == ActionTypes.ASSIGN_PROJECT:
            self._handle_assign_project(agent, action)
        elif action_type == ActionTypes.BUY_WARRIOR:
            self._handle_buy_warrior(agent, action['city_id'])
        elif action_type == ActionTypes.BUY_SETTLER:
            self._handle_buy_settler(agent, action['city_id'])
        
        # Update money and state
        self.money[agent] += self._calculate_gdp(agent)
        current_state = self._get_state_snapshot(agent)
        
        # Calculate rewards
        reward, components = self._calculate_reward(
            agent, prev_state, current_state
        )
        self.rewards[agent] = reward
        
        # Use episode_done method to check termination
        self.dones[agent] = self.episode_done()
        
        # Remove agent if done
        if self.dones[agent]:
            self.agents.remove(agent)
        
        # Update visit counts and select next agent
        self._update_state_visit_count(agent)
        if self.agents:
            self.agent_selection = self._agent_selector.next()
        else:
            self.agent_selection = None

    def _calculate_reward(
        self,
        agent: int,
        prev_state: Dict,
        current_state: Dict
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate reward and its components."""
        # Extract reward components
        components = {
            'P_progress': current_state['projects_in_progress'] - prev_state['projects_in_progress'],
            'P_completion': current_state['completed_projects'] - prev_state['completed_projects'],
            'C_tiles': current_state['explored_tiles'] - prev_state['explored_tiles'],
            'C_cities': self.cities_captured[agent],
            'L_cities': self.cities_lost[agent],
            'C_units': self.units_eliminated[agent],
            'L_units': self.units_lost[agent],
            'delta_GDP': current_state['gdp'] - prev_state['gdp'],
            'delta_Energy': current_state['energy_output'] - prev_state['energy_output'],
            'C_resources': self.resources_gained[agent],
            'E_impact': current_state['environmental_impact'],
            'Stalling': float(self._states_are_equal(prev_state, current_state)),
            'Entropy': calculate_entropy(self.state_visit_count[agent])
        }

        # Calculate total reward using weights
        reward = sum(
            REWARD_WEIGHTS[f'k{i+1}'] * value
            for i, value in enumerate(components.values())
        )
        
        return reward, components

    def reset(self, seed: Optional[int] = None) -> Dict:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        # Reset agents
        self.agents = self.possible_agents[:]
        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # Reset game state
        self._initialize_game_state()
        self._initialize_map()

        # Reset tracking variables
        self.rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        return {agent: self.observe(agent) for agent in self.agents}

    def observe(self, agent: int) -> Dict:
        """Get observation for an agent."""
        # Get masked map based on visibility
        masked_map = np.where(
            self.visibility_maps[agent][:, :, np.newaxis],
            self.map,
            np.zeros_like(self.map)
        )

        # Prepare unit observations
        units_obs = np.zeros(
            (self.max_units_per_agent, 4),
            dtype=np.float32
        )
        for idx, unit in enumerate(self.units[agent]):
            if idx < self.max_units_per_agent:
                units_obs[idx] = [
                    unit.x,
                    unit.y,
                    unit.health,
                    UNIT_TYPE_MAPPING[unit.type]
                ]

        # Get city observations
        cities_obs = self._get_agent_cities(agent)

        # Get money observation
        money_obs = np.array([self.money[agent]], dtype=np.float32)

        return {
            "map": masked_map,
            "units": units_obs,
            "cities": cities_obs,
            "money": money_obs
        }

    def render(self) -> None:
        """Render the current game state."""
        if self.render_mode is not None:
            game_state = {
                'map': self.map,
                'num_agents': self.num_of_agents,
                'visibility_maps': self.visibility_maps
            }
            self.renderer.render(game_state)

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'renderer'):
            self.renderer.close()

    def _get_state_snapshot(self, agent: int) -> Dict:
        """Get current state for an agent."""
        return {
            'projects_in_progress': len([
                city for city in self.cities[agent]
                if city.current_project is not None
            ]),
            'completed_projects': sum(
                len(city.completed_projects)
                for city in self.cities[agent]
            ),
            'explored_tiles': np.sum(self.visibility_maps[agent]),
            'cities_owned': len(self.cities[agent]),
            'units_owned': len(self.units[agent]),
            'gdp': self._calculate_gdp(agent),
            'energy_output': self._calculate_energy_output(agent),
            'resources_controlled': self._calculate_resources_controlled(agent),
            'environmental_impact': self.environmental_impact[agent]
        }

    def _update_state_visit_count(self, agent: int) -> None:
        """Update visit count for visible tiles."""
        visible_tiles = np.argwhere(self.visibility_maps[agent])
        for y, x in visible_tiles:
            tile_tuple = (x, y)
            if tile_tuple not in self.state_visit_count[agent]:
                self.state_visit_count[agent][tile_tuple] = 0
            self.state_visit_count[agent][tile_tuple] += 1

    def _handle_move_unit(self, agent: int, action: Dict) -> None:
        """Handle unit movement action."""
        unit_id = action['unit_id']
        if 0 <= unit_id < len(self.units[agent]):
            unit = self.units[agent][unit_id]
            unit.move(action['direction'])

    def _handle_attack_unit(self, agent: int, action: Dict) -> None:
        """Handle unit attack action."""
        unit_id = action['unit_id']
        if 0 <= unit_id < len(self.units[agent]):
            unit = self.units[agent][unit_id]
            unit.attack(action['direction'])

    def _handle_found_city(self, agent: int, action: Dict) -> None:
        """Handle city founding action."""
        unit_id = action['unit_id']
        if 0 <= unit_id < len(self.units[agent]):
            unit = self.units[agent][unit_id]
            if unit.type == 'settler' and unit.found_city():
                new_city = City(unit.x, unit.y, agent, self)
                self.cities[agent].append(new_city)
                self.units[agent].remove(unit)
                self._update_map_with_new_city(agent, new_city)

    def _handle_assign_project(self, agent: int, action: Dict) -> None:
        """Handle project assignment action."""
        city_id = action['city_id']
        if 0 <= city_id < len(self.cities[agent]):
            city = self.cities[agent][city_id]
            city.assign_project(action['project_id'])

    def _handle_buy_warrior(self, agent: int, city_id: int) -> None:
        """Handle warrior purchase action."""
        if 0 <= city_id < len(self.cities[agent]):
            city = self.cities[agent][city_id]
            if self.money[agent] >= WARRIOR_COST:
                self.money[agent] -= WARRIOR_COST
                placed = self._place_unit_near_city(agent, 'warrior', city.x, city.y)
                return placed
        return False

    def _handle_buy_settler(self, agent: int, city_id: int) -> None:
        """Handle settler purchase action."""
        if 0 <= city_id < len(self.cities[agent]):
            city = self.cities[agent][city_id]
            if self.money[agent] >= SETTLER_COST:
                self.money[agent] -= SETTLER_COST
                placed = self._place_unit_near_city(agent, 'settler', city.x, city.y)
                return placed
        return False

    def _place_unit_near_city(self, agent: int, unit_type: str, city_x: int, city_y: int) -> bool:
        """Place a new unit in an empty tile adjacent to a city."""
        adjacent_tiles = get_adjacent_tiles(city_x, city_y, self.map_width, self.map_height)
        for x, y in adjacent_tiles:
            if is_tile_empty(self.map, x, y, self.num_of_agents):
                self._place_unit(agent, unit_type, x, y)
                return True
        return False

    def _update_map_with_new_city(self, agent: int, city: City) -> None:
        """Update map state when a new city is founded."""
        x, y = city.x, city.y
        city_channel = self.num_of_agents + (3 * self.agents.index(agent))
        self.map[y, x, city_channel] = 1
        self._update_visibility(agent, x, y)
        self.map[y, x, self.agents.index(agent)] = 1

    def _process_city_projects(self, agent: int) -> None:
        """Process ongoing projects in all cities."""
        for city in self.cities[agent]:
            completed_project = city.process_project()
            if completed_project is not None:
                self._handle_completed_project(agent, city, completed_project)

    def _handle_completed_project(self, agent: int, city: City, project: Dict) -> None:
        """Handle completion of a project."""
        if project['type'] == 'unit':
            self._place_unit_near_city(agent, project['unit_type'], city.x, city.y)
        elif project['type'] in ['friendly', 'destructive']:
            gdp_boost = project.get('gdp_boost', 0)
            penalty = project.get('penalty', 0)
            self.gdp_bonus[agent] += gdp_boost
            self.environmental_impact[agent] += penalty
            if project['type'] == 'destructive':
                self._destroy_resource_near_city(city)

    def _destroy_resource_near_city(self, city: City) -> None:
        """Destroy a random resource near the city."""
        resource_channels_start = self.num_of_agents + 3 * self.num_of_agents
        resource_found = False

        # First check city tile
        for i in range(3):  # Check all resource types
            if self.map[city.y, city.x, resource_channels_start + i] > 0:
                self.map[city.y, city.x, resource_channels_start + i] = 0
                resource_found = True
                break

        # If no resource found on city tile, check adjacent tiles
        if not resource_found:
            adjacent_tiles = get_adjacent_tiles(city.x, city.y, self.map_width, self.map_height)
            for x, y in adjacent_tiles:
                for i in range(3):
                    if self.map[y, x, resource_channels_start + i] > 0:
                        self.map[y, x, resource_channels_start + i] = 0
                        return

    def _calculate_gdp(self, agent: int) -> float:
        """Calculate GDP for an agent."""
        return (
            len(self.cities[agent]) * 2 +
            self.gdp_bonus[agent]
        )

    def _calculate_energy_output(self, agent: int) -> float:
        """Calculate energy output for an agent."""
        return sum(city.resources['resource'] for city in self.cities[agent])

    def _calculate_resources_controlled(self, agent: int) -> float:
        """Calculate total resources controlled by an agent."""
        return sum(
            city.resources['material'] + city.resources['water']
            for city in self.cities[agent]
        )

    def _states_are_equal(self, state1: Dict, state2: Dict) -> bool:
        """Compare two states to check if they are identical."""
        if state1.keys() != state2.keys():
            return False
        
        for key in state1:
            if isinstance(state1[key], np.ndarray):
                if not np.array_equal(state1[key], state2[key]):
                    return False
            elif state1[key] != state2[key]:
                return False
        return True

    def _get_agent_cities(self, agent: int) -> np.ndarray:
        """Get city observations for an agent."""
        num_attributes = self._calculate_city_attributes()
        cities_obs = np.zeros((self.max_cities, num_attributes), dtype=np.float32)
        
        for idx, city in enumerate(self.cities[agent]):
            if idx >= self.max_cities:
                break
                
            city_data = [
                city.health,
                city.x,
                city.y,
                city.resources['resource'],
                city.resources['material'],
                city.resources['water']
            ]
            city_data.extend(city.completed_projects)
            city_data.append(city.current_project if city.current_project is not None else -1)
            city_data.append(city.project_duration)
            
            cities_obs[idx] = city_data[:num_attributes]
            
        return cities_obs