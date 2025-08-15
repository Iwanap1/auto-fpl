from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Player:
    pid: int
    name: str
    pos: str               # {"GK","DEF","MID","FWD"}
    team: str
    team_id: int
    price: float           
    features: Tuple # Last element must be availability
    pooling_metric: float  
    xP: float              
