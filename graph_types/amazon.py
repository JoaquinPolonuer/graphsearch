import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from pathlib import Path
from typing import Self

import pandas as pd

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
            name=str(row["brand_name"]),
            index=row["index"],
            type=row["type"],
        )


class ColorNode(Node):
    @classmethod
    def from_df_row(cls, row: pd.Series) -> Self:
        return cls(
            name=str(row["color_name"]),
            index=row["index"],
            type=row["type"],
        )


class CategoryNode(Node):
    @classmethod
    def from_df_row(cls, row: pd.Series) -> Self:
        return cls(
            name=str(row["category_name"]),
            index=row["index"],
            type=row["type"],
        )


if __name__ == "__main__":
    # Example usage
    mag_graph = Graph.load("amazon")
    node = mag_graph.get_node_by_index(0)
    second_hop = mag_graph.get_khop_idx(node, k=2)
