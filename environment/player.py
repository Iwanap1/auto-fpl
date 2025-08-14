from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class Player:
    pid: int
    name: str
    pos: str
    team: str
    team_id: int
    price: float  
    metric: float
    xP: float