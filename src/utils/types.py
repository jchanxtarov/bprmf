import enum
from typing import Dict, List

# dataset
UserItems = Dict[int, List[int]]


class ModelType(enum.Enum):
    BPRMF = "bprmf"
