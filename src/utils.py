import json
import os

import pandas as pd
import torch

from config import DATA_DIR
from graph_types.graph import Graph


def load_graph_and_qas(graph_name: str):
    qas = pd.read_csv(DATA_DIR / f"qas/{graph_name}.csv")
    graph = Graph.load(graph_name)
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


def iterate_qas(qas, limit=None, shuffle=False, subset=None):
    if subset is not None:
        if shuffle:
            raise ValueError("Cannot shuffle when subset is specified.")
        if limit is not None:
            raise ValueError("Cannot limit when subset is specified.")

        qas = qas[qas.index.isin(subset)]

    qas = qas.iloc[:limit] if limit else qas
    question_indices = qas.index.tolist()
    questions = qas["question"].tolist()
    answer_indices_list = qas["answer_indices"].apply(json.loads).tolist()

    if shuffle:
        indices = torch.randperm(len(question_indices)).tolist()
        question_indices = [question_indices[i] for i in indices]
        questions = [questions[i] for i in indices]
        answer_indices_list = [answer_indices_list[i] for i in indices]

    return list(zip(question_indices, questions, answer_indices_list))


def setup_results_dir(graph_name: str, experiment_name: str):
    results_dir = DATA_DIR / f"experiments/{graph_name}/{experiment_name}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
