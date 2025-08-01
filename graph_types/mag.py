import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from pathlib import Path
from typing import Self

import pandas as pd

from config import DATA_DIR

from graph_types.graph import Graph, Node


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

    @classmethod
    def from_df_row(cls, row: pd.Series) -> Self:
        return cls(
            name=str(row["title"]),
            abstract=row["abstract"],
            index=row["index"],
            type=row["type"],
        )


class AuthorNode(Node):
    @classmethod
    def from_df_row(cls, row: pd.Series) -> Self:
        return cls(
            name=str(row["DisplayName"]),
            index=row["index"],
            type=row["type"],
        )


class InstitutionNode(Node):
    @classmethod
    def from_df_row(cls, row: pd.Series) -> Self:
        return cls(
            name=str(row["DisplayName"]),
            index=row["index"],
            type=row["type"],
        )


class FieldOfStudyNode(Node):
    @classmethod
    def from_df_row(cls, row: pd.Series) -> Self:
        return cls(
            name=str(row["DisplayName"]),
            index=row["index"],
            type=row["type"],
        )


class MagGraph(Graph):

    @classmethod
    def load(cls) -> Self:
        return super().load(name="mag")

    def node_from_df_row(self, row: pd.Series) -> Node:
        if row["type"] == "author":
            return AuthorNode.from_df_row(row)
        elif row["type"] == "paper":
            return PaperNode.from_df_row(row)
        elif row["type"] == "institution":
            return InstitutionNode.from_df_row(row)
        elif row["type"] == "field_of_study":
            return FieldOfStudyNode.from_df_row(row)

        raise ValueError(f"Unknown node type: {row['type']}")

    def node_from_doc(self, doc: dict) -> Node:
        if doc["type"] == "paper":
            return PaperNode.from_doc(doc)
        elif doc["type"] == "author":
            return AuthorNode.from_doc(doc)
        elif doc["type"] == "institution":
            return InstitutionNode.from_doc(doc)
        elif doc["type"] == "field_of_study":
            return FieldOfStudyNode.from_doc(doc)
        else:
            raise ValueError(f"Unknown node type: {doc['type']}")


if __name__ == "__main__":
    # Example usage
    mag_graph = MagGraph.load()
    node = mag_graph.get_node_by_index(0)
    second_hop = mag_graph.get_khop_idx(node, k=2)
