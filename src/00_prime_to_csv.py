import os
import torch
import pickle
import pandas as pd
import json

edge_index = torch.load(f"data/00_raw_stark_graphs/prime/edge_index.pt")
edge_types = torch.load(f"data/00_raw_stark_graphs/prime/edge_types.pt")

with open(f"data/00_raw_stark_graphs/prime/edge_type_dict.pkl", "rb") as f:
    edge_type_dict = pickle.load(f)
    edge_types = [edge_type_dict[int(type)] for type in edge_types]

with open(f"data/00_raw_stark_graphs/prime/node_info.pkl", "rb") as f:
    node_info = pickle.load(f)

edges_df = pd.DataFrame(
    {"start_node_index": edge_index[0], "end_node_index": edge_index[1], "type": edge_types}
)
nodes_df = pd.DataFrame(node_info.values())
nodes_df["index"] = nodes_df.index
nodes_df["details"] = nodes_df["details"].apply(lambda x: json.dumps({} if pd.isna(x) else x))

os.makedirs("data/01_csv_graphs/prime/", exist_ok=True)
nodes_df.to_csv("data/01_csv_graphs/prime/nodes.csv", index=False)
edges_df.to_csv("data/01_csv_graphs/prime/edges.csv", index=False)

