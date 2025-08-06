import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from graph_types.graph import Graph, Node
from src.llms.simple_calls import (
    extract_entities_from_question,
    filter_relevant_nodes,
    answer_based_on_nodes,
)
from src.llms.agents.subgraph_explorer import SubgraphExplorerAgent


graph = Graph.load("prime")

question = "What are the reasons for the underuse of intranasal corticosteroids (INCSs) in the treatment of rhinitis?"

entities = extract_entities_from_question(question)

all_nodes = []
all_scores = []
for entity in entities:
    nodes, scores = graph.search_nodes(entity, k=3)
    all_nodes.extend(nodes)
    all_scores.extend(scores)

starting_nodes = filter_relevant_nodes(question, all_nodes, graph)

conversations_as_string = []
agent_answer_nodes: set[Node] = set()
for starting_node in starting_nodes:
    subgraph = graph.get_khop_subgraph(starting_node, k=2)
    agent = SubgraphExplorerAgent(
        node=starting_node,
        graph=subgraph,
        question=question,
    )
    agent_answer = set(agent.answer())
    agent_answer_nodes = agent_answer_nodes.union(agent_answer)
    conversations_as_string.append(agent.conversation_as_string)

answer = answer_based_on_nodes(
    question=question,
    nodes=list(agent_answer_nodes),
)

print(f"Question: {question}")
print(f"Agent answer nodes: {[node.name for node in agent_answer_nodes]}")
print(f"Agent answer: {answer}")
