import requests
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import requests
import sys
import time
from pathlib import Path
from config import DATA_DIR
import json
from typing import Optional, Self
import torch
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, field_validator
from src.graph_types.graph import Node, Graph
from src.keyword_search.index import ElasticsearchIndex


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
    def from_doc(cls, data: dict) -> Self:
        return cls(
            name=data["name"],
            source=data["source"],
            details=json.loads(data["details"]),
            index=data["index"],
            type=data["type"],
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

    def get_node_by_index(self, index: int) -> PrimeNode:
        node_row = self.nodes_df[self.nodes_df["index"] == index]

        if node_row.empty:
            raise ValueError(f"No node found with index {index}")

        row = node_row.iloc[0]
        return PrimeNode(
            name=row["name"],
            source=row["source"],
            details=json.loads(row["details"]),
            index=row["index"],
            type=row["type"],
            # semantic_embedding=row.get('semantic_embedding')
        )

    @classmethod
    def load(cls) -> Self:
        nodes_file = DATA_DIR / "01_csv_graphs/prime/nodes.csv"
        edges_file = DATA_DIR / "01_csv_graphs/prime/edges.csv"

        nodes_df = pd.read_csv(nodes_file)
        edges_df = pd.read_csv(edges_file)

        return cls(name="prime", nodes_df=nodes_df, edges_df=edges_df)

    def search_nodes(self, query: str, k=10) -> list[PrimeNode]:
        response = ElasticsearchIndex(name=f"{self.name}_index").search(query=query, k=k)
        hits = response.get("hits", {}).get("hits", [])
        return [PrimeNode.from_doc(hit["_source"]) for hit in hits]

    def get_neighbors(self, node: PrimeNode) -> set[PrimeNode]:
        neighbors_df = self.edges_df[self.edges_df["start_node_index"] == node.index]
        neighbor_indices = neighbors_df["end_node_index"].unique()
        return {self.get_node_by_index(idx) for idx in neighbor_indices}


if __name__ == "__main__":
    # Example usage
    graph = PrimeGraph.load()
    print(
        f"Loaded graph: {graph.name} with {len(graph.nodes_df)} nodes and {len(graph.edges_df)} edges"
    )

    # Example search
    results = graph.search_nodes("IL27")
    if results:
        print(f"Found {len(results)} nodes matching 'IL27':")
        for node in results:
            print(node)
    else:
        print("No nodes found matching 'IL27'")

    print(graph.get_neighbors(results[0]))
