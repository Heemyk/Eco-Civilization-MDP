from typing import Dict

# Action Types
class ActionTypes:
    MOVE_UNIT = 0
    ATTACK_UNIT = 1
    FOUND_CITY = 2
    ASSIGN_PROJECT = 3
    NO_OP = 4
    BUY_WARRIOR = 5
    BUY_SETTLER = 6

# Unit Types
UNIT_TYPE_MAPPING = {
    'warrior': 0,
    'settler': 1
}

# Resource Types
RESOURCE_CHANNELS = {
    'resource': 0,
    'material': 1,
    'water': 2
}

# Game Constants
MAX_AGENTS = 6
WARRIOR_COST = 40
SETTLER_COST = 60

# Reward Constants
REWARD_WEIGHTS = {
    'k1': 100.0,  # Progress of projects
    'k2': 200.0,  # Completion of projects
    'k3': 10.0,   # Tiles explored
    'k4': 500.0,  # Cities captured
    'k5': 0.0,    # Cities lost
    'k6': 100.0,  # Units eliminated
    'k7': 0.0,    # Units lost
    'k8': 50.0,   # Change in GDP
    'k9': 50.0,   # Change in Energy output
    'k10': 35.0,  # Resources gained
    'gamma': 0.00001,  # Environmental impact penalty
    'beta': 0.5,      # Stalling penalty
    'k_entropy': 0.4  # Entropy scaling
}

# Default Project Templates
DEFAULT_PROJECTS = {
    0: {'name': 'Make Warrior', 'duration': 3, 'type': 'unit', 'unit_type': 'warrior'},
    1: {'name': 'Make Settler', 'duration': 5, 'type': 'unit', 'unit_type': 'settler'}
}