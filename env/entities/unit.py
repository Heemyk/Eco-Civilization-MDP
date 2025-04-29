from typing import Optional, Tuple, Any

class Unit:
    """Represents a unit (warrior or settler) in the game."""
    
    def __init__(self, x: int, y: int, unit_type: str, owner: Any, env: Any):
        self.x = x
        self.y = y
        self.type = unit_type
        self.health = 100
        self.owner = owner
        self.env = env

    def move(self, direction: int) -> bool:
        """Move the unit in the specified direction."""
        new_pos = self._calculate_new_position(self.x, self.y, direction)
        if new_pos is not None:
            new_x, new_y = new_pos
            self.env._update_unit_position_on_map(self, new_x, new_y)
            self.x = new_x
            self.y = new_y
            return True
        return False

    def attack(self, direction: int) -> bool:
        """Attack in the specified direction."""
        if self.type != 'warrior':
            return False

        target_agent, target = self._check_enemy_units_and_cities(self.x, self.y, direction)
        if target is not None:
            target.health -= 35
            if target.health <= 0:
                self.env._remove_unit_or_city(target)
            return True
        return False

    def found_city(self) -> bool:
        """Found a city at current location if unit is a settler."""
        return self.type == 'settler'

    def _calculate_new_position(self, x: int, y: int, direction: int) -> Optional[Tuple[int, int]]:
        """Calculate new position based on direction."""
        delta_x, delta_y = 0, 0
        if direction == 0:  # up
            delta_y = -1
        elif direction == 1:  # right
            delta_x = 1
        elif direction == 2:  # down
            delta_y = 1
        elif direction == 3:  # left
            delta_x = -1

        new_x = x + delta_x
        new_y = y + delta_y

        if self._is_valid_move(new_x, new_y):
            return new_x, new_y
        return None

    def _is_valid_move(self, x: int, y: int) -> bool:
        """Check if move is valid."""
        if not (0 <= x < self.env.map_width and 0 <= y < self.env.map_height):
            return False
        return self._is_tile_empty_of_units_and_cities(x, y)

    def _is_tile_empty_of_units_and_cities(self, x: int, y: int) -> bool:
        """Check if tile is empty of units and cities."""
        for agent_idx in range(self.env.num_of_agents):
            unit_base_idx = self.env.num_of_agents + (3 * agent_idx)
            unit_channels = [unit_base_idx + i for i in range(3)]
            if any(self.env.map[y, x, channel] > 0 for channel in unit_channels):
                return False
        return True

    def _check_enemy_units_and_cities(self, x: int, y: int, direction: int) -> Tuple[Optional[Any], Optional[Any]]:
        """Check for enemy units or cities in attack range."""
        delta_x, delta_y = self._get_direction_deltas(direction)
        new_x, new_y = x + delta_x, y + delta_y

        if not self._is_valid_position(new_x, new_y):
            return None, None

        target = self.env._get_target_at(new_x, new_y)
        if target and target.owner != self.owner:
            return target.owner, target
        return None, None

    def _get_direction_deltas(self, direction: int) -> Tuple[int, int]:
        """Get x,y deltas for a direction."""
        deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
        return deltas[direction] if 0 <= direction < 4 else (0, 0)

    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is within map bounds."""
        return 0 <= x < self.env.map_width and 0 <= y < self.env.map_height