import json
import os
import pickle

import pandas as pd
import torch
import numpy as np

edge_index = torch.load(f"data/graphs/raw/amazon/edge_index.pt")
edge_types = torch.load(f"data/graphs/raw/amazon/edge_types.pt")

with open(f"data/graphs/raw/amazon/edge_type_dict.pkl", "rb") as f:
    print("Loading edge type dict...")
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
    f"data/graphs/raw/amazon/cache/brand-category-color/edge_index.pt"
)
edge_types_brand_category_color = torch.load(
    f"data/graphs/raw/amazon/cache/brand-category-color/edge_types.pt"
)
with open(
    f"data/graphs/raw/amazon/cache/brand-category-color/edge_type_dict.pkl",
    "rb",
) as f:
    print("Loading edge type dict for brand-category-color...")
    edge_type_dict_brand_category_color = pickle.load(f)
    edge_types_brand_category_color = [
        edge_type_dict_brand_category_color[int(type)] for type in edge_types_brand_category_color
    ]

edges_df_brand_category_color = pd.DataFrame(
    {
        "start_node_index": edge_index_brand_category_color[0],
        "end_node_index": edge_index_brand_category_color[1],
        "type": edge_types_brand_category_color,
    }
)
edges_df = pd.concat([edges_df, edges_df_brand_category_color], ignore_index=True).drop_duplicates()

del (
    edge_index_brand_category_color,
    edge_types_brand_category_color,
    edge_type_dict_brand_category_color,
)

print("Saving edges to CSV...")
os.makedirs("data/graphs/csv/amazon/", exist_ok=True)
edges_df.to_csv("data/graphs/csv/amazon/edges.csv", index=False)
del edges_df, edges_df_brand_category_color


with open(f"data/graphs/raw/amazon/node_info.pkl", "rb") as f:
    print("Loading node info...")
    node_info = pickle.load(f)
nodes_df = pd.DataFrame(node_info.values(), index=node_info.keys())
nodes_df["index"] = nodes_df.index
del node_info

with open(f"data/graphs/raw/amazon/cache/brand-category-color/node_info.pkl", "rb") as f:
    print("Loading node info for brand-category-color...")
    node_info_brand_category_color = pickle.load(f)
nodes_df_brand_category_color = pd.DataFrame(
    node_info_brand_category_color.values(), index=node_info_brand_category_color.keys()
)
nodes_df_brand_category_color["index"] = nodes_df_brand_category_color.index
del node_info_brand_category_color


nodes_df = pd.concat([nodes_df, nodes_df_brand_category_color], ignore_index=True).drop_duplicates(
    "index"
)


def clean_review_list(review_list):
    """Clean a list of review dictionaries to make it JSON serializable"""
    # Handle NaN or non-list values
    if not isinstance(review_list, list):
        return None

    cleaned_list = []

    for review_dict in review_list:
        cleaned = {}
        for key, value in review_dict.items():
            if pd.isna(value) or value is np.nan:
                cleaned[key] = None
            elif isinstance(value, (np.float64, np.float32)):
                cleaned[key] = float(value)
            elif isinstance(value, (np.int64, np.int32)):
                cleaned[key] = int(value)
            elif isinstance(value, np.bool_):
                cleaned[key] = bool(value)
            else:
                cleaned[key] = value
        cleaned_list.append(cleaned)

    return cleaned_list


print("Cleaning reviews...")
nodes_df["review"] = nodes_df["review"].apply(clean_review_list)

print("Cleaning qas")
nodes_df["qa"] = nodes_df["qa"].apply(clean_review_list)

print("Converting reviews to JSON strings...")
nodes_df["review"] = nodes_df["review"].apply(json.dumps)

print("Converting qas to JSON strings...")
nodes_df["qa"] = nodes_df["qa"].apply(json.dumps)

print("Saving nodes to CSV...")
nodes_df.to_csv("data/graphs/csv/amazon/nodes.csv", index=False)
