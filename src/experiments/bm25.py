import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import iterate_qas, load_embeddings, load_graph_and_qas, setup_results_dir
from src.experiments.utils import semantic_sort, map_entities_to_nodes, save_log, send_explorers
from graph_types.graph import Node

graph_name = "amazon"
graph, qas = load_graph_and_qas(graph_name)
results_dir = setup_results_dir(graph.name, "bm25")

for question_index, question, answer_indices in iterate_qas(qas, limit=1000):
    # tokenized_query = question.lower().split()
    bm25_nodes, scores = graph.search_nodes(question, k=100)
    save_log(
        {
            "question": question,
            "bm25_indices": [node.index for node in bm25_nodes],
            "answer_indices": answer_indices,
        },
        results_dir=results_dir,
        question_index=question_index,
    )

    print(f"Processed question {question_index}")
