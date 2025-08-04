import json
import os
import pickle

import pandas as pd
import torch

edge_index = torch.load(f"data/00_raw_stark_graphs/amazon/edge_index.pt")
edge_types = torch.load(f"data/00_raw_stark_graphs/amazon/edge_types.pt")

with open(f"data/00_raw_stark_graphs/amazon/edge_type_dict.pkl", "rb") as f:
    edge_type_dict = pickle.load(f)
    edge_types = [edge_type_dict[int(type)] for type in edge_types]

edges_df = pd.DataFrame(
    {
        "start_node_index": edge_index[0],
        "end_node_index": edge_index[1],
        "type": edge_types,
    }
)

del (
    edge_index,
    edge_types,
    edge_type_dict,
)

edge_index_brand_category_color = torch.load(
    f"data/00_raw_stark_graphs/amazon/cache/brand-category-color/edge_index.pt"
)
edge_types_brand_category_color = torch.load(
    f"data/00_raw_stark_graphs/amazon/cache/brand-category-color/edge_types.pt"
)
with open(
    f"data/00_raw_stark_graphs/amazon/cache/brand-category-color/edge_type_dict.pkl",
    "rb",
) as f:
    edge_type_dict_brand_category_color = pickle.load(f)
    edge_types_brand_category_color = [
        edge_type_dict_brand_category_color[int(type)]
        for type in edge_types_brand_category_color
    ]

edges_df_brand_category_color = pd.DataFrame(
    {
        "start_node_index": edge_index_brand_category_color[0],
        "end_node_index": edge_index_brand_category_color[1],
        "type": edge_types_brand_category_color,
    }
)
edges_df = pd.concat(
    [edges_df, edges_df_brand_category_color], ignore_index=True
).drop_duplicates()

del (
    edge_index_brand_category_color,
    edge_types_brand_category_color,
    edge_type_dict_brand_category_color,
)

os.makedirs("data/01_csv_graphs/amazon/", exist_ok=True)
edges_df.to_csv("data/01_csv_graphs/amazon/edges.csv", index=False)
del edges_df, edges_df_brand_category_color

with open(f"data/00_raw_stark_graphs/amazon/node_info.pkl", "rb") as f:
    print("Loading node info...")
    node_info = pickle.load(f)
nodes_df = pd.DataFrame(node_info.values(), index=node_info.keys())
nodes_df["index"] = nodes_df.index
del node_info

with open(
    f"data/00_raw_stark_graphs/amazon/cache/brand-category-color/node_info.pkl", "rb"
) as f:
    print("Loading node info for brand-category-color...")
    node_info_brand_category_color = pickle.load(f)
nodes_df_brand_category_color = pd.DataFrame(
    node_info_brand_category_color.values(), index=node_info_brand_category_color.keys()
)
nodes_df_brand_category_color["index"] = nodes_df_brand_category_color.index
del node_info_brand_category_color


nodes_df = pd.concat(
    [nodes_df, nodes_df_brand_category_color], ignore_index=True
).drop_duplicates("index")
nodes_df.to_csv("data/01_csv_graphs/amazon/nodes.csv", index=False)
