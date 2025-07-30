from typing import Self, Optional

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

    def node_from_doc(self, data: dict) -> Node:
        raise NotImplementedError("Subclasses must implement node_from_doc")

    def node_from_df_row(self, row: pd.Series) -> Node:
        raise NotImplementedError("Subclasses must implement node_from_df_row")

    def get_node_by_index(self, index: int) -> Node:
        node_row = self.nodes_df[self.nodes_df["index"] == index]

        if node_row.empty:
            raise ValueError(f"No node found with index {index}")

        row = node_row.iloc[0]
        return self.node_from_df_row(row)

    def get_neighbors_idx(self, node_index: int) -> pd.Series:
        neighbors_df_1 = self.edges_df[self.edges_df["start_node_index"] == node_index][
            "end_node_index"
        ]
        neighbors_df_2 = self.edges_df[self.edges_df["end_node_index"] == node_index][
            "start_node_index"
        ]
        neighbor_indices = pd.concat([neighbors_df_1, neighbors_df_2]).drop_duplicates()
        return neighbor_indices.reset_index(drop=True)

    def get_neighbors(self, node: Node) -> set[Node]:
        neighbor_indices = self.get_neighbors_idx(node.index)

        if len(neighbor_indices) > 50000:
            print(f"Lots of neighbors: {len(neighbor_indices)}. Returning empty set.")
            return set()

        return {self.get_node_by_index(idx) for idx in neighbor_indices}

    def search_nodes(self, query: str, k=10) -> list[Node]:
        from src.keyword_search.index import ElasticsearchIndex

        response = ElasticsearchIndex(name=f"{self.name}_index").search(query=query, k=k)
        hits = response.get("hits", {}).get("hits", [])
        return [self.node_from_doc(hit["_source"]) for hit in hits]

    def distance_to_all(self, node: Node, d: Optional[int] = None) -> pd.DataFrame:
        distances_df = pd.DataFrame({"dst_node_index": [node.index], "distance": [0]})
        current_level = pd.Series([node.index])
        current_distance = 0
        
        while len(current_level) > 0 and (d is None or current_distance < d):
            current_distance += 1
            next_level = []
            
            for node_idx in current_level:
                neighbor_indices = self.get_neighbors_idx(node_idx)
                # Filter out already visited nodes using vectorized operations
                unvisited_neighbors = neighbor_indices[~neighbor_indices.isin(distances_df["dst_node_index"])]
                
                if len(unvisited_neighbors) > 0:
                    # Create DataFrame for new distances and concat
                    new_distances = pd.DataFrame({
                        "dst_node_index": unvisited_neighbors,
                        "distance": current_distance
                    })
                    distances_df = pd.concat([distances_df, new_distances], ignore_index=True)
                    next_level.extend(unvisited_neighbors.tolist())
            
            current_level = pd.Series(next_level).drop_duplicates() if next_level else pd.Series([], dtype=int)
        
        return distances_df
