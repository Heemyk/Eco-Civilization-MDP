from typing import Dict, Any, Optional

class City:
    """Represents a city in the game."""
    
    def __init__(self, x: int, y: int, owner: Any, env: Any):
        self.x = x
        self.y = y
        self.health = 100
        self.type = "City"
        self.owner = owner
        self.env = env
        self.resources = self._get_resources()
        self.completed_projects = [0 for _ in range(self.env.max_projects)]
        self.current_project = None
        self.project_duration = 0

    def _get_resources(self) -> Dict[str, int]:
        """Calculate resources available to the city."""
        resources = {'resource': 0, 'material': 0, 'water': 0}
        scan_range = 2

        for dx in range(-scan_range, scan_range + 1):
            for dy in range(-scan_range, scan_range + 1):
                x, y = self.x + dx, self.y + dy
                if self._is_valid_position(x, y):
                    self._count_resources_at_tile(x, y, resources)
        
        return resources

    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is within map bounds."""
        return 0 <= x < self.env.map_width and 0 <= y < self.env.map_height

    def _count_resources_at_tile(self, x: int, y: int, resources: Dict[str, int]) -> None:
        """Count resources at a specific tile."""
        resource_channels_start = self.env.num_of_agents + 3 * self.env.num_of_agents
        resource_types = ['resource', 'material', 'water']
        
        for i, resource_type in enumerate(resource_types):
            if self.env.map[y, x, resource_channels_start + i] > 0:
                resources[resource_type] += 1

    def assign_project(self, project_id: int) -> bool:
        """Assign a new project to the city."""
        if self.current_project is not None:
            return False
            
        project = self.env.projects.get(project_id)
        if project is None:
            return False
            
        self.current_project = project_id
        self.project_duration = project['duration']
        return True

    def process_project(self) -> Optional[Dict]:
        """Process the current project and return it if completed."""
        if self.current_project is None:
            return None
            
        self.project_duration -= 1
        if self.project_duration <= 0:
            completed_project = self.env.projects[self.current_project]
            self.current_project = None
            return completed_project
            
        return None

    def get_state(self) -> Dict:
        """Get the current state of the city."""
        return {
            'position': (self.x, self.y),
            'health': self.health,
            'owner': self.owner,
            'resources': self.resources.copy(),
            'completed_projects': self.completed_projects.copy(),
            'current_project': self.current_project,
            'project_duration': self.project_duration
        }