from typing import Self

import pandas as pd
from fuzzywuzzy import fuzz
from pydantic import BaseModel


def fuzzy_match(name, pattern, threshold=90):
    return fuzz.partial_ratio(name.lower(), pattern.lower()) >= threshold


class Node(BaseModel):
    name: str
    index: int
    type: str

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, index={self.index}, type={self.type})"

    def __str__(self) -> str:
        return repr(self)

    def __hash__(self):
        return hash((self.name, self.index, self.type))

    def to_doc(self) -> dict:
        return {
            "name": self.name,
            "index": self.index,
            "type": self.type,
        }

    @classmethod
    def from_doc(cls, data: dict) -> Self:
        return cls(
            name=data["name"],
            index=data["index"],
            type=data["type"],
        )


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
        
    def get_node_by_index(self, index: int) -> Node:
        raise NotImplementedError("Subclasses must implement get_node_by_index")
