import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.algorithms import get_central_nodes_and_starting_node
from src.llms.simple_calls import (
    extract_entities_from_question,
    extract_question_answer_type,
)
from src.utils import (
    iterate_qas,
    load_embeddings,
    load_graph_and_qas,
    setup_results_dir,
)

graph_name = "prime"
doc_embeddings, query_embeddings = load_embeddings(graph_name)
graph, qas = load_graph_and_qas(graph_name)

results_dir = setup_results_dir(graph.name, "2hop")
for question_id, question, answer_indices in iterate_qas(qas, limit=100):

    question_embedding = query_embeddings[question_id][0]
    entities = extract_entities_from_question(question)

    sorted_central_nodes, starting_node = get_central_nodes_and_starting_node(
        graph, question, question_embedding, doc_embeddings, entities
    )

    candidates = list(graph.get_khop_idx(starting_node, k=2))

    if graph.name == "mag":
        answer_type = "paper"
    elif graph.name == "prime":
        answer_type = extract_question_answer_type(question, graph.node_types)
    elif graph.name == "amazon":
        answer_type = "product"

    candidates = graph.filter_indices_by_type(candidates, type=answer_type)

    sorted_candidates = sorted(
        candidates,
        key=lambda x: torch.matmul(
            question_embedding.detach().clone(), doc_embeddings[x].detach().clone().T
        ).item(),
        reverse=True,
    )

    log = {
        "question": question,
        "entities": [entity for entity in entities],
        "answer_type": answer_type,
        "starting_node_index": starting_node.index,
        "sorted_central_nodes_indices": [node.index for node in sorted_central_nodes],
        "sorted_candidates_indices": [int(i) for i in sorted_candidates],
        "answer_indices": answer_indices,
    }

    with open(results_dir / f"{question_id}.json", "w") as f:
        json.dump(log, f, indent=4)

    print(f"Processed question {question_id}")
