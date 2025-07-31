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

from config import DATA_DIR

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


class PrimeGraph(Graph):

    @classmethod
    def _check_columns(cls, v, required_columns, df_name="DataFrame"):
        actual_columns = set(v.columns)
        if actual_columns != required_columns:
            extra_columns = actual_columns - required_columns
            missing_columns = required_columns - actual_columns

            error_msg = f"{df_name} must have exactly these columns: {required_columns}"
            if missing_columns:
                error_msg += f". Missing: {missing_columns}"
            if extra_columns:
                error_msg += f". Extra: {extra_columns}"

            raise ValueError(error_msg)
        return v

    @field_validator("nodes_df", mode="before")
    @classmethod
    def validate_nodes_columns(cls, v):
        required_columns = {"name", "source", "details", "index", "type"}
        return cls._check_columns(v, required_columns, df_name="nodes DataFrame")

    @field_validator("edges_df", mode="before")
    @classmethod
    def validate_edges_columns(cls, v):
        required_columns = {"start_node_index", "end_node_index", "type"}
        return cls._check_columns(v, required_columns, df_name="edges DataFrame")

    @classmethod
    def load(cls) -> Self:
        nodes_file = DATA_DIR / "01_csv_graphs/prime/nodes.csv"
        edges_file = DATA_DIR / "01_csv_graphs/prime/edges.csv"

        nodes_df = pd.read_csv(nodes_file)
        edges_df = pd.read_csv(edges_file)

        return cls(name="prime", nodes_df=nodes_df, edges_df=edges_df)

    def node_from_doc(self, doc: dict) -> PrimeNode:
        return PrimeNode.from_doc(doc)

    def node_from_df_row(self, row: pd.Series) -> PrimeNode:
        return PrimeNode.from_df_row(row)


if __name__ == "__main__":
    # Example usage
    prime_graph = PrimeGraph.load()
    node = prime_graph.get_node_by_index(0)
    distance_df = prime_graph.get_khop_idx(node, k=1)
    print(distance_df)