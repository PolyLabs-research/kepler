"""
vocab.py - Token vocabulary
"""
import json
from pathlib import Path

class Vocabulary:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
    
    @property
    def vocab_size(self) -> int:
        return len(self.tokens)
    
    @property
    def pad_token(self) -> int:
        return self.token_to_id["PAD"]
    
    @property
    def bos_token(self) -> int:
        return self.token_to_id["BOS"]
    
    @property
    def eos_token(self) -> int:
        return self.token_to_id["EOS"]
    
    @property
    def unk_token(self) -> int:
        return self.token_to_id["UNK"]
    
    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id.get(t, self.unk_token) for t in tokens]
    
    def decode(self, ids: list[int]) -> list[str]:
        return [self.tokens[i] if 0 <= i < len(self.tokens) else "UNK" for i in ids]
    
    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"tokens": self.tokens}, f)
    
    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        with open(path) as f:
            return cls(json.load(f)["tokens"])
    
    @classmethod
    def default(cls) -> "Vocabulary":
        tokens = [
            "PAD", "BOS", "EOS", "UNK",
            # Digits & Ops
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "ADD", "SUB", "MUL", "DIV", "POW", "SQRT", "NEG", "POINT",
            # Functions
            "SIN", "COS", "TAN", "EXP", "LN", "LOG",
            # Variables
            "X", "T", "N", "VAR_1", "VAR_2", "VAR_3",
            # Constants (The new "Mathematical Brain")
            "CONST", "PI", "E", "PHI", "TAU", "SQRT2", "SQRT3",
            "HALF", "THIRD", "QUARTER",
            "INT_0", "INT_1", "INT_2", "INT_3", "INT_4", "INT_5", 
            "INT_6", "INT_7", "INT_8", "INT_9", "INT_10", "NEG_1", "NEG_2",
            # Physics (High value for your laws model)
            "M", "M1", "M2", "R", "F", "V", "P", "Q1", "Q2",
            "G_EARTH", "G_NEWTON", "C_LIGHT", "R_GAS",
            "DX_DT", "D2X_DT2", "EQ"
        ]
        return cls(tokens)