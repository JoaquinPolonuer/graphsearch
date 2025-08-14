import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from graph_types.graph import Graph, Node
from src.llms.simple_calls import (
    extract_entities_from_question,
    filter_relevant_nodes,
    answer_based_on_nodes,
)
from src.llms.agents.subgraph_explorer import SubgraphExplorerAgent
from logger_config import logger


graph = Graph.load("prime")

question = "What are common treatments for Tuberculosis in africa?"

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

answer = answer_based_on_nodes(
    question=question,
    nodes=list(agent_answer_nodes),
)

logger.info(f"Question: {question}")
logger.info(f"Agent answer nodes: {[node.name for node in agent_answer_nodes]}")
logger.info(f"Agent answer: {answer}")
