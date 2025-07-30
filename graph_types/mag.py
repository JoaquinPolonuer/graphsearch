import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from pathlib import Path
from typing import Self

import pandas as pd

from config import DATA_DIR
from .graph import Graph, Node


class PaperNode(Node):
    abstract: str

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


class AuthorNode(Node):
    pass


class InstitutionNode(Node):
    pass


class FieldOfStudyNode(Node):
    pass


class MagGraph(Graph):

    def get_node_by_index(self, index: int) -> Node:
        node_row = self.nodes_df[self.nodes_df["index"] == index]

        if node_row.empty:
            raise ValueError(f"No node found with index {index}")

        row = node_row.iloc[0]

        if row["type"] == "author":
            return AuthorNode(
                name=row["DisplayName"],
                index=row["index"],
                type=row["type"],
            )
        elif row["type"] == "paper":
            return PaperNode(
                name=row["title"],
                abstract=row["abstract"],
                index=row["index"],
                type=row["type"],
            )
        elif row["type"] == "institution":
            return InstitutionNode(
                name=row["DisplayName"],
                index=row["index"],
                type=row["type"],
            )
        elif row["type"] == "field_of_study":
            return FieldOfStudyNode(
                name=row["DisplayName"],
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
    graph = MagGraph.load()
    print(
        f"Loaded graph: {graph.name} with {len(graph.nodes_df)} nodes and {len(graph.edges_df)} edges"
    )

    # Example search
    print(graph.get_node_by_index(0))
    print(graph.get_node_by_index(1172724))
    print(graph.get_node_by_index(1104554))
    print(graph.get_node_by_index(1113255))
