import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import os
import torch
import pandas as pd
from src.utils import (
    iterate_qas,
    load_graph_and_qas,
    count_tokens,
    truncate_to_token_limit,
)
from openai import AzureOpenAI

graph_name = "prime"
graph, qas = load_graph_and_qas(graph_name)

AZURE_ENDPOINT = os.environ["AZURE_API_BASE"]
AZURE_API_KEY = os.environ["AZURE_API_KEY"]
AZURE_API_VERSION = os.environ["AZURE_API_VERSION"]

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
)

EMBEDDINGS_MODEL = "text-embedding-3-small"
EMBEDDINGS_DIR = f"data/graphs/embeddings/{EMBEDDINGS_MODEL}/{graph_name}"

node_embeddings = torch.load(f"{EMBEDDINGS_DIR}/node_embeddings.pt")


def search_nodes(query, top_k=5):
    response = client.embeddings.create(input=query, model=EMBEDDINGS_MODEL)
    query_embedding = torch.tensor(response.data[0].embedding)

    query_node_similarities = torch.matmul(query_embedding, node_embeddings.T)
    top_k_indices = torch.topk(query_node_similarities, k=top_k).indices.tolist()

    return [graph.get_node_by_index(i) for i in top_k_indices]


if __name__ == "__main__":
    results = search_nodes("POC1A", top_k=5)
    print(results)
