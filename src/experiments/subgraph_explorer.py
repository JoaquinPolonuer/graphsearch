import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from src.llms.simple_calls import extract_entities_from_question, filter_relevant_nodes
from src.utils import iterate_qas, load_embeddings, load_graph_and_qas, setup_results_dir
from src.experiments.utils import semantic_sort, map_entities_to_nodes, save_log, send_explorers
import argparse

parser = argparse.ArgumentParser(description="Generate embeddings for a graph.")
parser.add_argument("--graph_name", type=str, required=True, help="Name of the graph to process")
args = parser.parse_args()

graph_name = args.graph_name
node_embeddings, question_embeddings = load_embeddings(graph_name)
graph, qas = load_graph_and_qas(graph_name)

results_dir = setup_results_dir(graph.name, "subgraph_explorer")
for question_id, question, answer_indices in iterate_qas(qas, limit=100):
    if os.path.exists(results_dir / f"{question_id}.json"):
        print(f"Skipping {question_id} as it was already processed.")
        continue

    entities = extract_entities_from_question(question)

    all_nodes, all_scores = map_entities_to_nodes(graph, entities)
    starting_nodes = filter_relevant_nodes(question, all_nodes, graph)

    message_histories, agent_answer_indices = send_explorers(graph, question, starting_nodes)

    # NOTE: I'm commenting this out to be fair with prime. In the final version we might also want 
    # to filter by type for prime.
    
    # if graph.name == "mag":
    #     agent_answer_indices = graph.filter_indices_by_type(agent_answer_indices, "paper")
    # if graph.name == "amazon":
    #     agent_answer_indices = graph.filter_indices_by_type(agent_answer_indices, "product")

    # I'm not sure if we should do this. We should double check.
    similarities = question_embeddings[question_id] @ node_embeddings[agent_answer_indices].T
    agent_answer_indices = [
        x for _, x in sorted(zip(similarities.tolist(), agent_answer_indices), reverse=True)
    ]

    save_log(
        {
            "question": question,
            "all_nodes": [node.index for node in all_nodes],
            "message_histories": message_histories,
            "starting_nodes_indices": [node.index for node in starting_nodes],
            "agent_answer_indices": agent_answer_indices,
            "answer_indices": answer_indices,
        },
        results_dir=results_dir,
        question_id=question_id,
    )
