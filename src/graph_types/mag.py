import requests
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import time
from config import DATA_DIR
import json
from typing import Optional, Self
import torch
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, field_validator
from src.graph_types.graph import Node, Graph
from src.keyword_search.index import ElasticsearchIndex


class PaperNode(Node):
    name: str
    abstract: str

    def __repr__(self) -> str:
        return f"PaperNode(name={self.name}, index={self.index})"

    def to_doc(self) -> dict:
        return {
            "name": self.name,
            "abstract": self.abstract,
            "index": self.index,
            "type": self.type,
        }

    @classmethod
    def from_doc(cls, data: dict) -> Self:
        return cls(
            name=data["name"],
            abstract=data["abstract"],
            index=data["index"],
            type=data["type"],
        )

    def __hash__(self):
        return hash((self.name, self.abstract, self.index, self.type))


class AuthorNode(Node):
    name: str

    def __repr__(self) -> str:
        return f"AuthorNode(name={self.name}, index={self.index})"

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

    def __hash__(self):
        return hash((self.name, self.index, self.type))

class InstitutionNode(Node):
    name: str

    def __repr__(self) -> str:
        return f"InstitutionNode(name={self.name}, index={self.index})"

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

    def __hash__(self):
        return hash((self.name, self.index, self.type))
    
class FieldOfStudyNode(Node):
    name: str

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

    def __hash__(self):
        return hash((self.name, self.index, self.type))


class MagGraph(Graph):

    def get_node_by_index(self, index: int) -> Node:
        node_row = self.nodes_df[self.nodes_df["index"] == index]

        if node_row.empty:
            raise ValueError(f"No node found with index {index}")

        row = node_row.iloc[0]

        if row["type"] == "author":
            return AuthorNode(
                name=row["name"],
                index=row["index"],
                type=row["type"],
            )
        elif row["type"] == "paper":
            return PaperNode(
                name=row["name"],
                abstract=row["abstract"],
                index=row["index"],
                type=row["type"],
            )
        elif row["type"] == "institution":
            return InstitutionNode(
                name=row["name"],
                index=row["index"],
                type=row["type"],
            )
        elif row["type"] == "field_of_study":
            return FieldOfStudyNode(
                name=row["name"],
                index=row["index"],
                type=row["type"],
            )
        else:
            raise ValueError(f"Unknown node type: {row['type']}")

    @classmethod
    def load(cls) -> Self:
        nodes_file = DATA_DIR / "01_csv_graphs/mag/nodes.csv"
        edges_file = DATA_DIR / "01_csv_graphs/mag/edges.csv"

        nodes_df = pd.read_csv(nodes_file)
        edges_df = pd.read_csv(edges_file)

        return cls(name="mag", nodes_df=nodes_df, edges_df=edges_df)


if __name__ == "__main__":
    # Example usage
    graph = MagGraph.load()
    print(
        f"Loaded graph: {graph.name} with {len(graph.nodes_df)} nodes and {len(graph.edges_df)} edges"
    )

    # Example search
    graph.get_node_by_index(0)
    graph.get_node_by_index(100_000)

    graph.get_node_by_index(900_000)
    graph.get_node_by_index(1_800_000)
