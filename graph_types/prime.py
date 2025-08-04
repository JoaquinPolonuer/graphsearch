import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import sys
from pathlib import Path
from typing import Optional, Self

import pandas as pd
import torch
from pydantic import field_validator

from graph_types.graph import Graph, Node


class PrimeNode(Node):
    source: str
    details: dict
    semantic_embedding: Optional[torch.Tensor] = None

    class Config:
        arbitrary_types_allowed = True

    @field_validator("details", mode="before")
    @classmethod
    def validate_details(cls, v):
        if pd.isna(v):
            return {}
        return v

    def to_doc(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "details": json.dumps(self.details),
            "index": self.index,
            "type": self.type,
        }

    @classmethod
    def from_doc(cls, doc: dict) -> Self:
        return cls(
            name=doc["name"],
            source=doc["source"],
            details=json.loads(doc["details"]),
            index=doc["index"],
            type=doc["type"],
        )

    @classmethod
    def from_df_row(cls, row: pd.Series) -> Self:
        return cls(
            name=row["name"],
            source=row["source"],
            details=json.loads(row["details"]),
            index=row["index"],
            type=row["type"],
        )


if __name__ == "__main__":
    # Example usage
    prime_graph = Graph.load("prime")
    node = prime_graph.get_node_by_index(0)
    distance_df = prime_graph.get_khop_idx(node, k=1)
    print(distance_df)
