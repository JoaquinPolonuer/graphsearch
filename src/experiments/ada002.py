import torch
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import iterate_qas, load_embeddings, load_graph_and_qas, setup_results_dir
from src.experiments.utils import semantic_sort, map_entities_to_nodes, save_log, send_explorers
from graph_types.graph import Node

graph_name = "prime"
graph, qas = load_graph_and_qas(graph_name)
doc_embeddings, query_embeddings = load_embeddings(graph_name)
results_dir = setup_results_dir(graph.name, "ada002")

for question_index, question, answer_indices in iterate_qas(qas, limit=1000):
    question_embedding = query_embeddings[question_index]

    ada002_indices = semantic_sort(doc_embeddings.keys(), question_embedding, doc_embeddings)[:100]

    save_log(
        {
            "question": question,
            "ada002_indices": ada002_indices,
            "answer_indices": answer_indices,
        },
        results_dir=results_dir,
        question_index=question_index,
    )

    print(f"Processed question {question_index}")
