import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR
from llms.entity_extraction import (
    extract_entities_from_question,
    extract_question_answer_type,
    select_starting_node,
)
from utils import load_graph_and_qas, load_embeddings
from algorithms import mapped_nodes_by_relevance

graph_name = "prime"

graph, qas = load_graph_and_qas(graph_name)
doc_embeddings, query_embeddings = load_embeddings(graph.name)

results_dir = DATA_DIR / f"connectedness/{graph.name}_logs_2hop_filter_answer_type_and_starting_node"
os.makedirs(results_dir, exist_ok=True)
for question_index, row in qas.iloc[:1000].iterrows():
    question = row["question"]
    question_embedding = query_embeddings[question_index][0]
    answer_indices = json.loads(row["answer_indices"])

    entities = extract_entities_from_question(question)

    sorted_central_nodes = mapped_nodes_by_relevance(
        graph, question_embedding, doc_embeddings, entities
    )

    starting_node = select_starting_node(question, sorted_central_nodes) # sorted_central_nodes[0]

    candidates = list(graph.get_khop_idx(starting_node, k=2))

    if graph.name == "mag":
        answer_type = "paper"
    elif graph.name == "prime":
        answer_type = extract_question_answer_type(question, graph.node_types)

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

    with open(results_dir / f"{question_index}.json", "w") as f:
        json.dump(log, f, indent=4)

    print(f"Processed question {question_index}")
