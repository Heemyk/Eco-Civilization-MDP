from abc import ABC, abstractmethod
import pygame
import math
import numpy as np
from typing import Tuple, List, Dict

class BaseRenderer(ABC):
    """Abstract base class for renderers to ensure web compatibility."""
    
    @abstractmethod
    def render(self, game_state: Dict) -> None:
        """Render the current game state."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""
        pass

class PygameRenderer(BaseRenderer):
    """Pygame-based renderer implementation."""
    
    def __init__(self, map_width: int, map_height: int, cell_size: int = 40):
        pygame.init()
        self.cell_size = cell_size
        self.window_width = map_width * cell_size
        self.window_height = map_height * cell_size
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption('Civilization Environment')
        self.clock = pygame.time.Clock()
        
        # Color definitions
        self.agent_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255)   # Cyan
        ]
        
        self.resource_colors = {
            'resource': (200, 200, 200),   # Light gray
            'material': (139, 69, 19),     # Brown
            'water': (0, 191, 255)         # Deep sky blue
        }

    def render(self, game_state: Dict) -> None:
        """Render the current game state using Pygame."""
        self.screen.fill((0, 0, 0))  # Black background
        self._draw_grid()
        self._draw_map_elements(game_state)
        self._draw_visibility(game_state.get('visibility_maps', {}))
        pygame.display.flip()
        self.clock.tick(60)

    def _draw_grid(self) -> None:
        """Draw the grid lines."""
        for x in range(0, self.window_width, self.cell_size):
            pygame.draw.line(self.screen, (50, 50, 50), (x, 0), (x, self.window_height))
        for y in range(0, self.window_height, self.cell_size):
            pygame.draw.line(self.screen, (50, 50, 50), (0, y), (self.window_width, y))

    def _draw_map_elements(self, game_state: Dict) -> None:
        """Draw all map elements (resources, units, cities)."""
        map_data = game_state.get('map', None)
        if map_data is None:
            return

        # Draw ownership
        self._draw_territory_ownership(map_data, game_state.get('num_agents', 0))
        
        # Draw resources
        self._draw_resources(map_data, game_state.get('num_agents', 0))
        
        # Draw units and cities
        self._draw_units_and_cities(map_data, game_state.get('num_agents', 0))

    def _draw_territory_ownership(self, map_data: np.ndarray, num_agents: int) -> None:
        """Draw territory ownership for each agent."""
        height, width = map_data.shape[:2]
        for y in range(height):
            for x in range(width):
                for agent_idx in range(num_agents):
                    if map_data[y, x, agent_idx] == 1:
                        color = self.agent_colors[agent_idx % len(self.agent_colors)]
                        rect = pygame.Rect(
                            x * self.cell_size,
                            y * self.cell_size,
                            self.cell_size,
                            self.cell_size
                        )
                        pygame.draw.rect(self.screen, color, rect)
                        break

    def _draw_resources(self, map_data: np.ndarray, num_agents: int) -> None:
        """Draw resources on the map."""
        height, width = map_data.shape[:2]
        resource_start = num_agents + (3 * num_agents)
        
        for y in range(height):
            for x in range(width):
                for i, (resource, color) in enumerate(self.resource_colors.items()):
                    if map_data[y, x, resource_start + i] > 0:
                        self._draw_circle(x, y, color)

    def _draw_units_and_cities(self, map_data: np.ndarray, num_agents: int) -> None:
        """Draw units and cities for each agent."""
        height, width = map_data.shape[:2]
        
        for agent_idx in range(num_agents):
            unit_base_idx = num_agents + (3 * agent_idx)
            color = self.agent_colors[agent_idx % len(self.agent_colors)]
            
            for y in range(height):
                for x in range(width):
                    # Draw city
                    if map_data[y, x, unit_base_idx] > 0:
                        self._draw_star(x, y, self._darken_color(color))
                    # Draw warrior
                    if map_data[y, x, unit_base_idx + 1] > 0:
                        self._draw_triangle(x, y, color)
                    # Draw settler
                    if map_data[y, x, unit_base_idx + 2] > 0:
                        self._draw_square(x, y, color)

    def _draw_visibility(self, visibility_maps: Dict) -> None:
        """Draw visibility overlays for each agent."""
        agent_shades = [
            (255, 0, 0, 50),    # Red with alpha
            (0, 255, 0, 50),    # Green with alpha
            (0, 0, 255, 50),    # Blue with alpha
            (255, 255, 0, 50),  # Yellow with alpha
            (255, 0, 255, 50),  # Magenta with alpha
            (0, 255, 255, 50)   # Cyan with alpha
        ]

        for agent_idx, visibility_map in visibility_maps.items():
            shade_surface = pygame.Surface(
                (self.cell_size, self.cell_size),
                pygame.SRCALPHA
            )
            shade_surface.fill(agent_shades[agent_idx % len(agent_shades)])
            visible_tiles = np.argwhere(visibility_map)
            
            for y, x in visible_tiles:
                self.screen.blit(
                    shade_surface,
                    (x * self.cell_size, y * self.cell_size)
                )

    def _draw_circle(self, x: int, y: int, color: Tuple[int, int, int]) -> None:
        """Draw a circle (resource)."""
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 4
        pygame.draw.circle(self.screen, color, (center_x, center_y), radius)

    def _draw_square(self, x: int, y: int, color: Tuple[int, int, int]) -> None:
        """Draw a square (settler)."""
        padding = self.cell_size // 8
        rect = pygame.Rect(
            x * self.cell_size + padding,
            y * self.cell_size + padding,
            self.cell_size - 2 * padding,
            self.cell_size - 2 * padding
        )
        pygame.draw.rect(self.screen, color, rect)

    def _draw_triangle(self, x: int, y: int, color: Tuple[int, int, int]) -> None:
        """Draw a triangle (warrior)."""
        half_size = self.cell_size // 2
        quarter_size = self.cell_size // 4
        center_x = x * self.cell_size + half_size
        center_y = y * self.cell_size + half_size
        
        points = [
            (center_x, center_y - quarter_size),
            (center_x - quarter_size, center_y + quarter_size),
            (center_x + quarter_size, center_y + quarter_size)
        ]
        pygame.draw.polygon(self.screen, color, points)

    def _draw_star(self, x: int, y: int, color: Tuple[int, int, int]) -> None:
        """Draw a star (city)."""
        center_x = x * self.cell_size + self.cell_size // 2
        center_y = y * self.cell_size + self.cell_size // 2
        radius_outer = self.cell_size // 3
        radius_inner = self.cell_size // 6
        num_points = 5
        points = []
        
        for i in range(num_points * 2):
            angle = i * math.pi / num_points - math.pi / 2
            radius = radius_outer if i % 2 == 0 else radius_inner
            px = center_x + radius * math.cos(angle)
            py = center_y + radius * math.sin(angle)
            points.append((px, py))
            
        pygame.draw.polygon(self.screen, color, points)

    def _darken_color(self, color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Create a darker version of a color."""
        return tuple(max(0, min(255, int(c * 0.7))) for c in color)

    def close(self) -> None:
        """Clean up Pygame resources."""
        pygame.quit()

class WebRenderer(BaseRenderer):
    """Placeholder for web-based renderer."""
    
    def __init__(self):
        self.game_states = []

    def render(self, game_state: Dict) -> None:
        """Store game state for web rendering."""
        self.game_states.append(game_state)

    def close(self) -> None:
        """Clean up web renderer resources."""
        self.game_states.clear()

    def get_game_states(self) -> List[Dict]:
        """Get recorded game states for web playback."""
        return self.game_states