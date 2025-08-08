import pickle

import pandas as pd
import torch

nodes_df = pd.read_parquet("data/graphs/parquet/mag/nodes.parquet")

name_column = {
    "paper": "title",
    "author": "DisplayName",
    "institution": "DisplayName",
    "field_of_study": "DisplayName",
}

nodes_df["name"] = nodes_df.apply(lambda row: row[name_column[row["type"]]], axis=1)
nodes_df["summary"] = nodes_df["abstract"].fillna("")

nodes_df.to_parquet("data/graphs/parquet/mag/nodes.parquet")
