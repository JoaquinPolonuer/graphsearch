import torch
import pandas as pd
from typing import Optional
from pydantic import BaseModel

from fuzzywuzzy import fuzz


def fuzzy_match(name, pattern, threshold=90):
    return fuzz.partial_ratio(name.lower(), pattern.lower()) >= threshold


class Node(BaseModel):
    # name: str
    index: int
    type: str
    # semantic_embedding: Optional[torch.Tensor] = None

    def __repr__(self) -> str:
        return f"Node(name={self.name}, index={self.index}, type={self.type})"

    def __str__(self) -> str:
        return repr(self)


class Edge(BaseModel):
    start_node_index: int
    end_node_index: int
    type: str

    def __repr__(self):
        return f"Edge(start_node={self.start_node_index}, type={self.type}, end_node={self.end_node_index})"

    def __str__(self):
        return repr(self)


class Graph(BaseModel):
    name: str
    nodes_df: pd.DataFrame
    edges_df: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True
