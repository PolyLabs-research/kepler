import numpy as np
from abc import ABC, abstractmethod
from core.constants import snap_to_token

def float_to_tokens(v: float) -> list[str]:
    """Smart tokenizer: Snaps to constants first, then falls back to digits."""
    snapped = snap_to_token(v)
    if snapped:
        return [snapped]
    
    tokens = ["CONST"]
    if v < 0:
        tokens.append("NEG")
        v = abs(v)
    
    formatted = f"{v:.2f}"
    for char in formatted:
        if char == ".": tokens.append("POINT")
        else: tokens.append(char)
    return tokens

class BaseGenerator(ABC):
    SUBTYPES = {}
    WEIGHTS = {}
    
    def _sample_subtype(self) -> str:
        subtypes = list(self.SUBTYPES.keys())
        weights = np.array([self.WEIGHTS.get(s, 1.0) for s in subtypes])
        return np.random.choice(subtypes, p=weights / weights.sum())

    def _float_to_tokens(self, v: float) -> list[str]:
        return float_to_tokens(v)

    @abstractmethod
    def generate(self, subtype: str = None, tier: int = 1) -> tuple[np.ndarray, list[str], str]:
        """
        Generate a single example.
        
        Args:
            subtype: Specific subtype to generate (None for random)
            tier: Difficulty tier (1=easy, 2=medium, 3=hard)
        
        Returns:
            Tuple of (features, tokens, actual_subtype)
        """
        pass
