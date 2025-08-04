import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from llms.simple_calls import extract_entities_from_question, filter_relevant_nodes
from utils import iterate_qas, load_embeddings, load_graph_and_qas, setup_results_dir
from llms.agents.subgraph_explorer import SubgraphExplorerAgent
from graph_types.graph import Node

graph_name = "prime"
doc_embeddings, query_embeddings = load_embeddings(graph_name)
graph, qas = load_graph_and_qas(graph_name)

results_dir = setup_results_dir(graph.name, "2hop")
for question_index, question, answer_indices in list(iterate_qas(qas, limit=1000))[1:]:

    entities = extract_entities_from_question(question)

    all_nodes = []
    all_scores = []
    for entity in entities:
        nodes, scores = graph.search_nodes(entity, k=3)
        all_nodes.extend(nodes)
        all_scores.extend(scores)

    starting_nodes = filter_relevant_nodes(question, all_nodes, graph)

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
        
    agent_answer_indices = [node.index for node in agent_answer_nodes]

    print(agent_answer_nodes)
