import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from pathlib import Path
from typing import Self

import pandas as pd

from config import DATA_DIR

from graph_types.graph import Graph, Node


class ProductNode(Node):
    @classmethod
    def from_df_row(cls, row: pd.Series) -> Self:
        return cls(
            name=str(row["title"]),
            index=row["index"],
            type=row["type"],
        )


class BrandNode(Node):
    @classmethod
    def from_df_row(cls, row: pd.Series) -> Self:
        return cls(
            name=str(row["title"]),
            index=row["index"],
            type=row["type"],
        )


class ColorNode(Node):
    @classmethod
    def from_df_row(cls, row: pd.Series) -> Self:
        return cls(
            name=str(row["title"]),
            index=row["index"],
            type=row["type"],
        )


class CategoryNode(Node):
    @classmethod
    def from_df_row(cls, row: pd.Series) -> Self:
        return cls(
            name=str(row["title"]),
            index=row["index"],
            type=row["type"],
        )


NODE_TYPE_MAPPING = {
    "product": ProductNode,
    "brand": BrandNode,
    "color": ColorNode,
    "category": CategoryNode,
}


class AmazonGraph(Graph):

    @classmethod
    def load(cls) -> Self:
        return super().load(name="amazon")

    def get_node_class_by_type(self, node_type: str) -> type[Node]:
        if node_type in NODE_TYPE_MAPPING:
            return NODE_TYPE_MAPPING[node_type]
        raise ValueError(f"Unknown node type: {node_type}")

    def node_from_df_row(self, row: pd.Series) -> Node:
        return self.get_node_class_by_type(row["type"]).from_df_row(row)

    def node_from_doc(self, doc: dict) -> Node:
        return self.get_node_class_by_type(doc["type"]).from_doc(doc)


if __name__ == "__main__":
    # Example usage
    mag_graph = AmazonGraph.load()
    node = mag_graph.get_node_by_index(0)
    second_hop = mag_graph.get_khop_idx(node, k=2)
