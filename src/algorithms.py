from difflib import SequenceMatcher

import torch

from graph_types.graph import Graph, Node


def normalized_edit_distance(name: str, question: str) -> float:
    name_lower = name.lower()
    question_lower = question.lower()

    if name_lower in question_lower:
        return 0.0

    matcher = SequenceMatcher(None, name_lower, question_lower)
    similarity = matcher.ratio()

    edit_distance_ratio = 1 - similarity

    return edit_distance_ratio


def get_central_nodes_and_starting_node(
    graph: Graph,
    question: str,
    question_embedding: torch.Tensor,
    doc_embeddings: torch.Tensor,
    entities: list[str],
) -> tuple[list[Node], Node]:

    all_nodes = []
    all_scores = []
    for entity in entities:
        nodes, scores = graph.search_nodes(entity, k=1)
        all_nodes.extend(nodes)
        all_scores.extend(scores)

    if graph.name == "prime":
        sorted_nodes = sorted(
            all_nodes,
            key=lambda x: torch.matmul(
                question_embedding.detach().clone(),
                doc_embeddings[x.index].detach().clone().T,
            ).item(),
            reverse=True,
        )
    elif graph.name in ["mag", "amazon"]:
        sorted_nodes = [
            node
            for _, node in sorted(
                zip(all_scores, all_nodes), key=lambda x: x[0], reverse=True
            )
        ]

    # Sort nodes by normalized edit distance (lower is better, so no reverse)
    # sorted_nodes = sorted(
    #     all_nodes,
    #     key=lambda node: (
    #         normalized_edit_distance(node.name, question),
    #         -len(node.name)  # negative for descending order (longest name first)
    #     )
    # )

    starting_node = sorted_nodes[0]

    return sorted_nodes, starting_node
