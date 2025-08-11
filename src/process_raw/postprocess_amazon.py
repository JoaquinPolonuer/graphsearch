import pickle

import pandas as pd
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.process_raw.utils import add_summary_amazon


nodes_df = pd.read_parquet("data/graphs/parquet/amazon/nodes.parquet")

node_types_brand_category_color = torch.load(
    f"data/graphs/raw/amazon/cache/brand-category-color/node_types.pt"
)
with open(f"data/graphs/raw/amazon/cache/brand-category-color/node_type_dict.pkl", "rb") as f:
    node_types_dict_brand_category_color = pickle.load(f)
    node_types_brand_category_color = [
        node_types_dict_brand_category_color[int(type)] for type in node_types_brand_category_color
    ]
nodes_df["type"] = node_types_brand_category_color

print("Adding names...")
name_column = {
    "product": "title",
    "brand": "brand_name",
    "color": "color_name",
    "category": "category_name",
}

nodes_df["name"] = nodes_df.apply(lambda row: row[name_column[row["type"]]], axis=1)

print("Adding summaries...")
nodes_df["summary"] = nodes_df.apply(lambda row: add_summary_amazon(row), axis=1)

print("Saving nodes...")
nodes_df.to_parquet("data/graphs/parquet/amazon/nodes.parquet")
