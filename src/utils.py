import json
import os

import pandas as pd
import torch
import tiktoken

from config import DATA_DIR
from graph_types.graph import Graph


def load_graph_and_qas(graph_name: str):
    qas = pd.read_csv(DATA_DIR / f"qas/stark_{graph_name}_qa.csv")
    graph = Graph.load(graph_name)
    return graph, qas


def load_embeddings(graph_name: str):
    node_embeddings = torch.load(
        DATA_DIR / f"graphs/embeddings/text-embedding-3-small/{graph_name}/node_embeddings.pt"
    )
    question_embeddings = torch.load(
        DATA_DIR / f"graphs/embeddings/text-embedding-3-small/{graph_name}/question_embeddings.pt"
    )
    return node_embeddings, question_embeddings


def iterate_qas(qas, limit=None):
    question_ids = qas["id"].tolist()
    questions = qas["query"].tolist()
    answer_indices_list = qas["answer_ids"].apply(json.loads).tolist()

    if limit:
        return list(zip(question_ids, questions, answer_indices_list))[:limit]
    return list(zip(question_ids, questions, answer_indices_list))


def setup_results_dir(graph_name: str, experiment_name: str):
    results_dir = DATA_DIR / f"experiments/{graph_name}/{experiment_name}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    """Count tokens in text using tiktoken for OpenAI models."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def truncate_to_token_limit(
    text: str, max_tokens: int = 8192, model: str = "text-embedding-3-small"
) -> str:
    """Truncate text to fit within token limit."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)
