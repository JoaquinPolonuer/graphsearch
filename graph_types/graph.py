from typing import Optional, Self

import pandas as pd
from fuzzywuzzy import fuzz
from pydantic import BaseModel, field_validator


def fuzzy_match(name, pattern, threshold=90):
    return fuzz.partial_ratio(name.lower(), pattern.lower()) >= threshold


class Node(BaseModel):
    name: str
    index: int
    type: str
    summary: str

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
            "summary": self.summary,
        }

    @classmethod
    def from_doc(cls, data: dict) -> Self:
        return cls(
            name=data["name"],
            index=data["index"],
            type=data["type"],
            summary=data.get("summary", ""),
        )

    @classmethod
    def from_df_row(cls, row: pd.Series) -> Self:
        return cls(
            name=row["name"],
            index=row["index"],
            type=row["type"],
            summary=row.get("summary", ""),
        )


class Edge(BaseModel):
    start_node_index: int
    end_node_index: int
    type: str

    def __repr__(self):
        return f"Edge(start_node={self.start_node_index}, type={self.type}, end_node={self.end_node_index})"

    def __str__(self):
        return repr(self)


class Path(BaseModel):
    path_as_list: list[Node | str]

    @field_validator("path_as_list", mode="before")
    @classmethod
    def validate_path_as_list(cls, path_as_list):
        for i in range(0, len(path_as_list), 2):
            assert isinstance(
                path_as_list[i], Node
            ), f"Expected Node at index {i}, got {type(path_as_list[i])}"
        for i in range(1, len(path_as_list), 2):
            assert isinstance(
                path_as_list[i], str
            ), f"Expected str at index {i}, got {type(path_as_list[i])}"
        return path_as_list

    def __hash__(self):
        # A frozenset is an immutable, unordered collection of unique elements.
        # Here, we use it to ensure that the hash is the same for a path and its reverse,
        # since {path, reversed_path} as a frozenset will be equal for both.
        fwd = tuple(self.path_as_list)
        rev = tuple(reversed(self.path_as_list))
        return hash(frozenset([fwd, rev]))

    def __eq__(self, other):
        if not isinstance(other, Path):
            return False
        fwd_self = tuple(self.path_as_list)
        rev_self = tuple(reversed(self.path_as_list))
        fwd_other = tuple(other.path_as_list)
        rev_other = tuple(reversed(other.path_as_list))
        return frozenset([fwd_self, rev_self]) == frozenset([fwd_other, rev_other])

    def __repr__(self):
        return " <-> ".join(
            [str(node) if isinstance(node, Node) else node for node in self.path_as_list]
        )

    def __str__(self):
        return repr(self)


class Graph(BaseModel):

    name: str
    nodes_df: pd.DataFrame
    edges_df: pd.DataFrame
    index: Optional["TantivyIndex"] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        from src.keyword_search.index import TantivyIndex

        super().__init__(**data)

        self.index = TantivyIndex(name=f"{self.name}_index")

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
        neighbor_indices = (
            pd.concat(
                [
                    self.edges_df[self.edges_df["start_node_index"] == node_index][
                        "end_node_index"
                    ],
                    self.edges_df[self.edges_df["end_node_index"] == node_index][
                        "start_node_index"
                    ],
                ]
            )
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

        return set(neighbor_indices.values)

    def get_neighbors(self, node: Node) -> set[Node]:
        neighbor_indices = self.get_neighbors_idx(node.index)

        if len(neighbor_indices) > 50000:
            print(f"Lots of neighbors: {len(neighbor_indices)}. Returning empty set.")
            return set()

        return {self.get_node_by_index(idx) for idx in neighbor_indices}

    def get_khop_idx(self, node: Node, k: int) -> set[int]:
        first_hop_neighbors = self.get_neighbors_idx(node.index)
        if k == 1:
            return first_hop_neighbors

        if k == 2:
            second_hop_neighbors = pd.Series(dtype=int)

            first_hop_neighbors = pd.Series(list(first_hop_neighbors), name="neighbor_index")
            neighbors_df_1 = (
                self.edges_df[self.edges_df["start_node_index"].isin(first_hop_neighbors)][
                    "end_node_index"
                ]
                .rename("neighbor_index_1")
                .drop_duplicates()
            )
            neighbors_df_2 = (
                self.edges_df[self.edges_df["end_node_index"].isin(first_hop_neighbors)][
                    "start_node_index"
                ]
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

    def get_khop_subgraph(self, node: Node, k: int) -> Self:
        neighbors_idx = self.get_khop_idx(node, k)
        nodes_df = self.nodes_df[self.nodes_df["index"].isin(neighbors_idx)]
        edges_df = self.edges_df[
            self.edges_df["start_node_index"].isin(neighbors_idx)
            | self.edges_df["end_node_index"].isin(neighbors_idx)
        ]
        return self.__class__(
            name=f"{k}-hop of {self.name} around {node.name}",
            nodes_df=nodes_df,
            edges_df=edges_df,
        )

    def search_nodes(self, query: str, k=10, mode="default") -> tuple[list[Node], list[float]]:
        if self.index is None:
            return [], []
            
        if mode == "default":
            response = self.index.search(query=query, k=k)
        elif mode == "summary":
            response = self.index.search_summary(query=query, k=k)
        else:
            raise ValueError(f"Unsupported search mode: {mode}")
        hits = response.get("hits", {}).get("hits", [])
        return [self.node_from_doc(hit["_source"]) for hit in hits], [hit["_score"] for hit in hits]

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

    def simple_search_in_surroundings(
        self,
        node: Node,
        query: Optional[str] = None,
        type: Optional[str] = None,
        k: int = 1,
    ) -> list[Node]:
        subgraph = self.get_khop_subgraph(node, k)
        search_nodes_df = subgraph.nodes_df

        if type:
            search_nodes_df = search_nodes_df[search_nodes_df["type"] == type]

        if query:
            search_nodes_df = search_nodes_df[
                (search_nodes_df["name"].apply(lambda x: fuzzy_match(x, query)))
                | (search_nodes_df["name"].str.contains(query, case=False))
            ]

        if search_nodes_df.empty:
            return []

        return [self.get_node_by_index(idx) for idx in search_nodes_df["index"].tolist()]

    # NOTE: There's a bug with undirected graphs here!!!! We are not considering this
    # A -> B <- C
    def find_paths_of_length_2(self, src: Node, dst: Node):
        if src.index == dst.index:
            return []

        direct_connections = pd.concat(
            [
                self.edges_df[
                    (
                        (self.edges_df["start_node_index"] == src.index)
                        & (self.edges_df["end_node_index"] == dst.index)
                    )
                ].rename({"start_node_index": "src_index", "end_node_index": "dst_index"}, axis=1),
                self.edges_df[
                    (
                        (self.edges_df["start_node_index"] == dst.index)
                        & (self.edges_df["end_node_index"] == src.index)
                    )
                ].rename({"start_node_index": "dst_index", "end_node_index": "src_index"}, axis=1),
            ]
        )

        src_mediators = self.edges_df[(self.edges_df["start_node_index"] == src.index)].rename(
            {
                "start_node_index": "src_index",
                "end_node_index": "neighbor",
            },
            axis=1,
        )
        mediator_src = self.edges_df[(self.edges_df["end_node_index"] == src.index)].rename(
            {
                "start_node_index": "neighbor",
                "end_node_index": "src_index",
            },
            axis=1,
        )
        src_edges = (
            pd.concat([src_mediators, mediator_src]).drop_duplicates().reset_index(drop=True)
        )

        mediator_dst = self.edges_df[(self.edges_df["end_node_index"] == dst.index)].rename(
            {
                "start_node_index": "neighbor",
                "end_node_index": "dst_index",
            },
            axis=1,
        )
        dst_mediators = self.edges_df[(self.edges_df["start_node_index"] == dst.index)].rename(
            {
                "start_node_index": "dst_index",
                "end_node_index": "neighbor",
            },
            axis=1,
        )

        dst_edges = (
            pd.concat([mediator_dst, dst_mediators]).drop_duplicates().reset_index(drop=True)
        )

        paths_with_mediator = (
            pd.merge(
                src_edges,
                dst_edges,
                left_on="neighbor",
                right_on="neighbor",
                suffixes=("_1", "_2"),
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )

        paths = set()

        for _, row in direct_connections.iterrows():
            path = Path(
                path_as_list=[
                    self.get_node_by_index(row["src_index"]),
                    row["type"],
                    self.get_node_by_index(row["dst_index"]),
                ]
            )
            paths.add(path)

        for _, row in paths_with_mediator.iterrows():
            path = Path(
                path_as_list=[
                    self.get_node_by_index(row["src_index"]),
                    row["type_1"],
                    self.get_node_by_index(row["neighbor"]),
                    row["type_2"],
                    self.get_node_by_index(row["dst_index"]),
                ]
            )
            paths.add(path)

        return paths
