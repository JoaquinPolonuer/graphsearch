import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import sys
from pathlib import Path
from typing import Optional, Self

import pandas as pd
import torch
from pydantic import field_validator

from graph_types.graph import Graph, Node


class PrimeNode(Node):
    pass


if __name__ == "__main__":
    # Example usage
    prime_graph = Graph.load("prime")
    node = prime_graph.get_node_by_index(0)
    distance_df = prime_graph.get_khop_idx(node, k=1)
    print(distance_df)
