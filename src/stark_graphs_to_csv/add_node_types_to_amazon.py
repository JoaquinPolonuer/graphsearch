import torch
import pickle
import pandas as pd

edges_df = pd.read_csv("data/01_csv_graphs/amazon/edges.csv", low_memory=False)
nodes_df = pd.read_csv("data/01_csv_graphs/amazon/nodes.csv", low_memory=False)

node_types_brand_category_color = torch.load(
    f"data/00_raw_stark_graphs/amazon/cache/brand-category-color/node_types.pt"
)
with open(
    f"data/00_raw_stark_graphs/amazon/cache/brand-category-color/node_type_dict.pkl", "rb"
) as f:
    node_types_dict_brand_category_color = pickle.load(f)
    node_types_brand_category_color = [
        node_types_dict_brand_category_color[int(type)] for type in node_types_brand_category_color
    ]
nodes_df["type"] = node_types_brand_category_color

nodes_df.drop(columns="review").to_csv("data/01_csv_graphs/amazon/nodes_reduced.csv", index=False)
