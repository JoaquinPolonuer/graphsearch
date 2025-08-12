import json
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from graph_types.graph import Node
from src.llms.agents.subgraph_explorer import SubgraphExplorerAgent


def semantic_sort(indices, query_embedding, doc_embeddings):
    return sorted(
        indices,
        key=lambda x: torch.matmul(
            query_embedding.detach().clone(),
            doc_embeddings[x].detach().clone().T,
        ).item(),
        reverse=True,
    )


def send_explorers(graph, question, starting_nodes):
    message_histories = []
    agent_answer_nodes: set[Node] = set()
    for starting_node in starting_nodes:
        subgraph = graph.get_khop_subgraph(starting_node, k=2)
        agent = SubgraphExplorerAgent(
            node=starting_node,
            graph=subgraph,
            question=question,
        )
        for selected_tool, answer_nodes in agent.answer():
            pass

        agent_answer_nodes = agent_answer_nodes.union(set(answer_nodes))
        message_histories.append(agent.message_history)

    agent_answer_indices = [node.index for node in agent_answer_nodes]
    return message_histories, agent_answer_indices


def map_entities_to_nodes(graph, entities):
    all_nodes = []
    all_scores = []
    for entity in entities:
        nodes, scores = graph.search_nodes(entity, k=1)
        all_nodes.extend(nodes)
        all_scores.extend(scores)
    return all_nodes, all_scores


def save_log(log, results_dir, question_id):
    with open(results_dir / f"{question_id}.json", "w") as f:
        json.dump(log, f, indent=4)
