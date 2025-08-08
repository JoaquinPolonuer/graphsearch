import pickle

import pandas as pd
import torch

nodes_df = pd.read_parquet("data/graphs/parquet/prime/nodes.parquet")
nodes_df["summary"] = nodes_df["details"].fillna("")

nodes_df.to_parquet("data/graphs/parquet/prime/nodes.parquet")
