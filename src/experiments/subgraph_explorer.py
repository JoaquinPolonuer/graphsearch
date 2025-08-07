import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from graph_types.graph import Node
from src.llms.simple_calls import extract_entities_from_question, filter_relevant_nodes
from src.llms.agents.subgraph_explorer import SubgraphExplorerAgent
from src.utils import iterate_qas, load_embeddings, load_graph_and_qas, setup_results_dir

graph_name = "amazon"
doc_embeddings, query_embeddings = load_embeddings(graph_name)
graph, qas = load_graph_and_qas(graph_name)

results_dir = setup_results_dir(graph.name, "subgraph_explorer")
for question_index, question, answer_indices in list(iterate_qas(qas, limit=1000, shuffle=True))[
    :100
]:

    if os.path.exists(results_dir / f"{question_index}.json"):
        # with open(results_dir / f"{question_index}.json", "r") as f:
        #     log = json.load(f)
        # if set(log.get("agent_answer_indices")).issuperset(set(log.get("answer_indices"))):
        #     print(f"Skipping {question_index} as it was correctly solved.")
        #     continue
        # else:
        #     print(f"Re-solving {question_index} as it was not correctly solved.")
        print(f"Skipping {question_index} as it was already processed.")
        continue

    entities = extract_entities_from_question(question)

    all_nodes = []
    all_scores = []
    for entity in entities:
        nodes, scores = graph.search_nodes(entity, k=3)
        all_nodes.extend(nodes)
        all_scores.extend(scores)

    starting_nodes = filter_relevant_nodes(question, all_nodes, graph)

    message_histories = []
    agent_answer_nodes: set[Node] = set()
    for starting_node in starting_nodes:
        subgraph = graph.get_khop_subgraph(starting_node, k=2)

        # subgraph.nodes_df["similarity"] = subgraph.nodes_df["index"].apply(
        #     lambda x: torch.matmul(
        #         query_embeddings[question_index].detach().clone(),
        #         doc_embeddings[x].detach().clone().T,
        #     ).item()
        # )

        # subgraph.nodes_df = subgraph.nodes_df.sort_values(by="similarity", ascending=False)

        agent = SubgraphExplorerAgent(
            node=starting_node,
            graph=subgraph,
            question=question,
        )
        for selected_tool, final_answer in agent.answer():
            if final_answer is not None:
                agent_answer_nodes = agent_answer_nodes.union(set(final_answer))
                break
        message_histories.append(agent.message_history)

    agent_answer_indices = [node.index for node in agent_answer_nodes]

    if graph.name == "mag":
        agent_answer_indices = graph.filter_indices_by_type(agent_answer_indices, "paper")
    if graph.name == "amazon":
        agent_answer_indices = graph.filter_indices_by_type(agent_answer_indices, "product")

    agent_answer_indices = sorted(
        agent_answer_indices,
        key=lambda x: torch.matmul(
            query_embeddings[question_index].detach().clone(),
            doc_embeddings[x].detach().clone().T,
        ).item(),
        reverse=True,
    )

    log = {
        "question": question,
        "all_nodes": [node.index for node in all_nodes],
        "message_histories": message_histories,
        "starting_nodes_indices": [node.index for node in starting_nodes],
        "agent_answer_indices": agent_answer_indices,
        "answer_indices": answer_indices,
    }

    with open(results_dir / f"{question_index}.json", "w") as f:
        json.dump(log, f, indent=4)
