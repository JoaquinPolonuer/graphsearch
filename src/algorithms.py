import torch
from graph_types.graph import Graph, Node


def mapped_nodes_by_relevance(
    graph: Graph,
    question_embedding: torch.Tensor,
    doc_embeddings: torch.Tensor,
    entities: list[str],
) -> list[Node]:

    all_nodes = []
    all_scores = []
    for entity in entities:
        nodes, scores = graph.search_nodes(entity, k=3)
        all_nodes.extend(nodes)
        all_scores.extend(scores)

    if graph.name == "prime":
        sorted_nodes = sorted(
            all_nodes,
            key=lambda x: torch.matmul(
                question_embedding.detach().clone(), doc_embeddings[x.index].detach().clone().T
            ).item(),
            reverse=True,
        )
    elif graph.name == "mag":
        sorted_nodes = [
            node for _, node in sorted(zip(all_scores, all_nodes), key=lambda x: x[0], reverse=True)
        ]

    return sorted_nodes
