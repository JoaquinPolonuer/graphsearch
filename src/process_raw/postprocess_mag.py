import pickle

import pandas as pd
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.process_raw.utils import add_summary_to_mag

nodes_df = pd.read_parquet("data/graphs/parquet/mag/nodes.parquet")

print("Adding names...")
name_column = {
    "paper": "title",
    "author": "DisplayName",
    "institution": "DisplayName",
    "field_of_study": "DisplayName",
}

nodes_df["name"] = nodes_df.apply(lambda row: row[name_column[row["type"]]], axis=1)

print("Adding summaries...")
nodes_df["summary"] = nodes_df.apply(lambda row: add_summary_to_mag(row), axis=1)

print("Saving nodes...")
nodes_df.to_parquet("data/graphs/parquet/mag/nodes.parquet")
