from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Player:
    pid: int
    name: str
    pos: str               # {"GK","DEF","MID","FWD"}
    team: str
    team_id: int
    price: float           # £m
    features: Tuple[float, float, float, float, float]  # n=5 where index 4 is prob of playing
    pooling_metric: float  # for candidate sampling/ranking
    xP: float              # expected points
