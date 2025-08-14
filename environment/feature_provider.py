from player import Player
from typing import Callable, List, Optional
import numpy as np

class FeatureProvider:
    def __init__(self, fn: Optional[Callable[[List[Player], int], np.ndarray]] = None):
        self.fn = fn

    def get_factors(self, players: List[Player], gw: int) -> np.ndarray:
        
        # returns nx5 array of factors for each player

    def get_availability()
