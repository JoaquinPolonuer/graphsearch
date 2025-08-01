import torch
import pickle
import pandas as pd

nodes_df = pd.read_parquet("data/graphs/parquet/amazon/nodes.parquet")

node_types_brand_category_color = torch.load(
    f"data/graphs/raw/amazon/cache/brand-category-color/node_types.pt"
)
with open(
    f"data/graphs/raw/amazon/cache/brand-category-color/node_type_dict.pkl", "rb"
) as f:
    node_types_dict_brand_category_color = pickle.load(f)
    node_types_brand_category_color = [
        node_types_dict_brand_category_color[int(type)] for type in node_types_brand_category_color
    ]
nodes_df["type"] = node_types_brand_category_color

nodes_df.to_parquet("data/graphs/parquet/amazon/nodes.parquet")
