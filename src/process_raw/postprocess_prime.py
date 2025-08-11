import pickle

import pandas as pd
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.process_raw.utils import add_summary_to_prime

nodes_df = pd.read_parquet("data/graphs/parquet/prime/nodes.parquet")
nodes_df["summary"] = nodes_df.apply(lambda row: add_summary_to_prime(row), axis=1)

nodes_df.to_parquet("data/graphs/parquet/prime/nodes.parquet")
