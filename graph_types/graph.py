from typing import Optional, Self

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

    @classmethod
    def from_df_row(cls, row: pd.Series) -> Self:
        return cls(
            name=row["name"],
            index=row["index"],
            type=row["type"],
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

    @property
    def node_types(self) -> list[str]:
        if not hasattr(self, "_node_types_cache"):
            self._node_types_cache = list(self.nodes_df["type"].unique())
        return self._node_types_cache

    @classmethod
    def load(cls, name) -> Self:
        from config import DATA_DIR

        nodes_file = DATA_DIR / f"graphs/parquet/{name}/nodes.parquet"
        edges_file = DATA_DIR / f"graphs/parquet/{name}/edges.parquet"

        nodes_df = pd.read_parquet(nodes_file)
        edges_df = pd.read_parquet(edges_file)

        return cls(name=name, nodes_df=nodes_df, edges_df=edges_df)

    def get_node_class_by_type(self, node_type: str) -> type[Node]:
        from config import NODE_TYPE_MAPPING

        if node_type in NODE_TYPE_MAPPING:
            return NODE_TYPE_MAPPING[node_type]
        raise ValueError(f"Unknown node type: {node_type}")

    def node_from_df_row(self, row: pd.Series) -> Node:
        return self.get_node_class_by_type(row["type"]).from_df_row(row)

    def node_from_doc(self, doc: dict) -> Node:
        return self.get_node_class_by_type(doc["type"]).from_doc(doc)

    def get_node_by_index(self, index: int) -> Node:
        node_row = self.nodes_df[self.nodes_df["index"] == index]

        if node_row.empty:
            raise ValueError(f"No node found with index {index}")

        row = node_row.iloc[0]
        return self.node_from_df_row(row)

    def get_neighbors_idx(self, node_index: int) -> set[int]:
        neighbors_df_1 = self.edges_df[self.edges_df["start_node_index"] == node_index][
            "end_node_index"
        ].rename("neighbor_index_1")
        neighbors_df_2 = self.edges_df[self.edges_df["end_node_index"] == node_index][
            "start_node_index"
        ].rename("neighbor_index_2")
        neighbor_indices = (
            pd.concat([neighbors_df_1, neighbors_df_2])
            .drop_duplicates()
            .rename("neighbor_index")
            .reset_index(drop=True)
        )

        if self.name == "mag":
            neighbor_indices = pd.merge(
                neighbor_indices.reset_index(drop=True),
                self.nodes_df[["index", "type"]],
                left_on="neighbor_index",
                right_on="index",
            )
            return set(
                neighbor_indices[neighbor_indices["type"] != "field_of_study"][
                    "neighbor_index"
                ].values
            )
        else:
            return set(neighbor_indices.values)

    def get_khop_idx(self, node: Node, k: int) -> set[int]:
        first_hop_neighbors = self.get_neighbors_idx(node.index)
        if k == 1:
            return first_hop_neighbors

        if k == 2:
            second_hop_neighbors = pd.Series(dtype=int)

            first_hop_neighbors = pd.Series(
                list(first_hop_neighbors), name="neighbor_index"
            )
            neighbors_df_1 = (
                self.edges_df[
                    self.edges_df["start_node_index"].isin(first_hop_neighbors)
                ]["end_node_index"]
                .rename("neighbor_index_1")
                .drop_duplicates()
            )
            neighbors_df_2 = (
                self.edges_df[
                    self.edges_df["end_node_index"].isin(first_hop_neighbors)
                ]["start_node_index"]
                .rename("neighbor_index_2")
                .drop_duplicates()
            )

            second_hop_neighbors = (
                pd.concat([first_hop_neighbors, neighbors_df_1, neighbors_df_2])
                .drop_duplicates()
                .reset_index(drop=True)
            )

            return set(second_hop_neighbors.values)
        raise ValueError(f"Unsupported value for k: {k}. Only 1 or 2 are supported.")

    def get_neighbors(self, node: Node) -> set[Node]:
        neighbor_indices = self.get_neighbors_idx(node.index)

        if len(neighbor_indices) > 50000:
            print(f"Lots of neighbors: {len(neighbor_indices)}. Returning empty set.")
            return set()

        return {self.get_node_by_index(idx) for idx in neighbor_indices}

    def search_nodes(self, query: str, k=10) -> tuple[list[Node], list[float]]:
        from src.keyword_search.index import ElasticsearchIndex

        response = ElasticsearchIndex(name=f"{self.name}_index").search(
            query=query, k=k
        )
        hits = response.get("hits", {}).get("hits", [])
        return [self.node_from_doc(hit["_source"]) for hit in hits], [
            hit["_score"] for hit in hits
        ]

    def filter_indices_by_type(self, indices: list[int], type: str) -> list[int]:
        indices_df = pd.DataFrame(indices, columns=["index"])
        indices_df = pd.merge(
            indices_df,
            self.nodes_df[["index", "type"]],
            on="index",
            how="left",
        )
        filtered_indices = indices_df[indices_df["type"] == type]["index"].tolist()
        return filtered_indices
