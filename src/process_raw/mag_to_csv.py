import json
import os
import pickle

import pandas as pd
import torch

edge_index = torch.load(f"data/graphs/raw/mag/edge_index.pt")
edge_types = torch.load(f"data/graphs/raw/mag/edge_types.pt")

with open(f"data/graphs/raw/mag/edge_type_dict.pkl", "rb") as f:
    edge_type_dict = pickle.load(f)
    edge_types = [edge_type_dict[int(type)] for type in edge_types]

with open(f"data/graphs/raw/mag/node_info.pkl", "rb") as f:
    node_info = pickle.load(f)

edges_df = pd.DataFrame(
    {
        "start_node_index": edge_index[0],
        "end_node_index": edge_index[1],
        "type": edge_types,
    }
)
nodes_df = pd.DataFrame(node_info.values())
nodes_df["index"] = nodes_df.index

os.makedirs("data/graphs/csv/mag/", exist_ok=True)
nodes_df.to_csv("data/graphs/csv/mag/nodes.csv", index=False)
edges_df.to_csv("data/graphs/csv/mag/edges.csv", index=False)
