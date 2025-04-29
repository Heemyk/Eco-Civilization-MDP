from typing import List, Tuple, Dict
import numpy as np

def get_adjacent_tiles(x: int, y: int, map_width: int, map_height: int) -> List[Tuple[int, int]]:
    """
    Get a list of valid adjacent tile coordinates.
    
    Args:
        x (int): X coordinate
        y (int): Y coordinate
        map_width (int): Width of the map
        map_height (int): Height of the map
        
    Returns:
        List[Tuple[int, int]]: List of valid adjacent coordinates
    """
    adjacent_coords = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            adj_x, adj_y = x + dx, y + dy
            if 0 <= adj_x < map_width and 0 <= adj_y < map_height:
                adjacent_coords.append((adj_x, adj_y))
    return adjacent_coords

def calculate_visibility(
    x: int,
    y: int,
    visibility_range: int,
    map_width: int,
    map_height: int
) -> np.ndarray:
    """
    Calculate visibility map for a given position and range.
    
    Args:
        x (int): X coordinate of the unit/city
        y (int): Y coordinate of the unit/city
        visibility_range (int): Number of tiles visible in each direction
        map_width (int): Width of the map
        map_height (int): Height of the map
        
    Returns:
        np.ndarray: Boolean array indicating visible tiles
    """
    visibility = np.zeros((map_height, map_width), dtype=bool)
    
    x_min = max(0, x - visibility_range)
    x_max = min(map_width, x + visibility_range + 1)
    y_min = max(0, y - visibility_range)
    y_max = min(map_height, y + visibility_range + 1)
    
    visibility[y_min:y_max, x_min:x_max] = True
    return visibility

def place_resources(
    map_data: np.ndarray,
    num_agents: int,
    bountifulness: float = 0.15
) -> np.ndarray:
    """
    Place resources randomly on the map.
    
    Args:
        map_data (np.ndarray): The game map
        num_agents (int): Number of agents in the game
        bountifulness (float): Proportion of tiles that should have resources
        
    Returns:
        np.ndarray: Updated map with resources placed
    """
    height, width = map_data.shape[:2]
    num_resources = int(bountifulness * height * width)
    resource_channels_start = num_agents + 3 * num_agents

    # Get all possible tile positions
    all_tiles = [(x, y) for x in range(width) for y in range(height)]
    np.random.shuffle(all_tiles)

    resources_placed = 0
    tile_index = 0

    while resources_placed < num_resources and tile_index < len(all_tiles):
        x, y = all_tiles[tile_index]
        tile_index += 1

        # Check if tile already has resources
        if np.any(map_data[y, x, resource_channels_start:resource_channels_start + 3] > 0):
            continue

        # Place random resource type
        resource_type = np.random.choice(['resource', 'material', 'water'])
        if resource_type == 'resource':
            map_data[y, x, resource_channels_start] = 1
        elif resource_type == 'material':
            map_data[y, x, resource_channels_start + 1] = 1
        else:  # water
            map_data[y, x, resource_channels_start + 2] = 1

        resources_placed += 1

    return map_data

def calculate_entropy(visit_counts: Dict[Tuple[int, int], int]) -> float:
    """
    Calculate Shannon entropy of visited states.
    
    Args:
        visit_counts (Dict[Tuple[int, int], int]): Dictionary mapping tile coordinates to visit counts
        
    Returns:
        float: Entropy value
    """
    if not visit_counts:
        return 0.0
        
    total_visits = sum(visit_counts.values())
    probabilities = [count / total_visits for count in visit_counts.values()]
    entropy = -sum(p * np.log(p + 1e-12) for p in probabilities)
    return entropy

def is_tile_empty(
    map_data: np.ndarray,
    x: int,
    y: int,
    num_agents: int
) -> bool:
    """
    Check if a tile is empty (no units or cities).
    
    Args:
        map_data (np.ndarray): The game map
        x (int): X coordinate to check
        y (int): Y coordinate to check
        num_agents (int): Number of agents in the game
        
    Returns:
        bool: True if tile is empty, False otherwise
    """
    # Check ownership channels
    if np.any(map_data[y, x, :num_agents] > 0):
        return False

    # Check unit and city channels
    unit_channels_start = num_agents
    unit_channels_end = unit_channels_start + (3 * num_agents)
    if np.any(map_data[y, x, unit_channels_start:unit_channels_end] > 0):
        return False

    return True

def get_tile_info(
    map_data: np.ndarray,
    x: int,
    y: int,
    num_agents: int
) -> Dict:
    """
    Get detailed information about a tile.
    
    Args:
        map_data (np.ndarray): The game map
        x (int): X coordinate
        y (int): Y coordinate
        num_agents (int): Number of agents in the game
        
    Returns:
        Dict: Information about the tile contents
    """
    tile_info = {
        'ownership': None,
        'units': [],
        'cities': [],
        'resources': []
    }

    # Check ownership
    for agent_idx in range(num_agents):
        if map_data[y, x, agent_idx] > 0:
            tile_info['ownership'] = agent_idx
            break

    # Check units and cities
    for agent_idx in range(num_agents):
        unit_base_idx = num_agents + (3 * agent_idx)
        if map_data[y, x, unit_base_idx] > 0:
            tile_info['cities'].append(agent_idx)
        if map_data[y, x, unit_base_idx + 1] > 0:
            tile_info['units'].append(('warrior', agent_idx))
        if map_data[y, x, unit_base_idx + 2] > 0:
            tile_info['units'].append(('settler', agent_idx))

    # Check resources
    resource_channels_start = num_agents + 3 * num_agents
    resource_types = ['resource', 'material', 'water']
    for i, resource_type in enumerate(resource_types):
        if map_data[y, x, resource_channels_start + i] > 0:
            tile_info['resources'].append(resource_type)

    return tile_info