# agents/base.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
from enum import Enum

class ActionType(Enum):
    MOVE_UNIT = "move_unit"
    ATTACK_UNIT = "attack_unit"
    FOUND_CITY = "found_city"
    ASSIGN_PROJECT = "assign_project"
    NO_OP = "no_op"
    BUY_WARRIOR = "buy_warrior"
    BUY_SETTLER = "buy_settler"

class Action(BaseModel):
    action_type: ActionType
    unit_id: Optional[int]
    direction: Optional[int]
    city_id: Optional[int]
    project_id: Optional[int]
    reasoning: str

class Observation(BaseModel):
    map_state: str
    visible_units: str
    visible_cities: str
    resources: str
    current_money: float

class Memory(BaseModel):
    short_term: List[Dict[str, str]]
    long_term: List[Dict[str, str]]
    last_reward: float
    last_action: Optional[Action]