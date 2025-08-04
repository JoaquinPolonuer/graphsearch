import os
import json
import torch
import pandas as pd
from config import DATA_DIR
from graph_types.prime import PrimeGraph
from graph_types.mag import MagGraph
from graph_types.amazon import AmazonGraph


def load_graph_and_qas(graph_name: str):
    qas = pd.read_csv(DATA_DIR / f"qas/{graph_name}.csv")
    if graph_name == "prime":
        graph = PrimeGraph.load()
    elif graph_name == "mag":
        graph = MagGraph.load()
    elif graph_name == "amazon":
        graph = AmazonGraph.load()
    return graph, qas


def load_embeddings(graph_name: str):
    doc_embeddings = torch.load(
        DATA_DIR
        / f"graphs/embeddings/{graph_name}/text-embedding-ada-002/doc/candidate_emb_dict.pt"
    )
    query_embeddings = torch.load(
        DATA_DIR / f"graphs/embeddings/{graph_name}/text-embedding-ada-002/query/query_emb_dict.pt"
    )
    return doc_embeddings, query_embeddings


def iterate_qas(qas, limit=None):
    qas = qas.iloc[:limit] if limit else qas
    question_indices = qas.index.tolist()
    questions = qas["question"].tolist()
    answer_indices_list = qas["answer_indices"].apply(json.loads).tolist()
    return zip(question_indices, questions, answer_indices_list)


def setup_results_dir(graph_name: str, experiment_name: str):
    results_dir = DATA_DIR / f"experiments/{graph_name}/{experiment_name}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
