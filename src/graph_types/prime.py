import json
from typing import Optional
import torch
import pandas as pd
from pydantic import field_validator
from src.graph_types.graph import Node, Graph


class PrimeNode(Node):
    id: str
    name: str
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

    def __repr__(self) -> str:
        return f"Node(name={self.name}, index={self.index}, type={self.type})"

    def __str__(self) -> str:
        return repr(self)


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
        required_columns = {"id", "name", "source", "details", "index", "type"}
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
            id=str(row["id"]),
            name=row["name"],
            source=row["source"],
            details=json.loads(row["details"]),
            index=row["index"],
            type=row["type"],
            # semantic_embedding=row.get('semantic_embedding')
        )
