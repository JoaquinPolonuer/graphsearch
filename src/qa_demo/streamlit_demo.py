import streamlit as st
import time
import random
from collections import deque

import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from graph_types.graph import Graph, Node
from src.llms.simple_calls import (
    extract_entities_from_question,
    filter_relevant_nodes,
    answer_based_on_nodes,
)
from src.llms.agents.subgraph_explorer import SubgraphExplorerAgent

graph = Graph.load("prime")

# Phrase variations for dynamic UI
SEARCH_PHRASES = [
    "Discovering",
    "Locating",
    "Scanning for",
    "Exploring",
    "Identifying",
    "Retrieving",
    "Tracking down",
    "Pinpointing",
    "Investigating",
]

CONNECTION_PHRASES = [
    "Analyzing the relationship between",
    "Exploring connections linking",
    "Mapping pathways from",
    "Investigating how",
    "Tracing connections between",
    "Examining the bridge connecting",
    "Discovering links between",
    "Unraveling associations from",
    "Probing the network between",
    "Navigating from",
]


st.set_page_config(page_title="g1 prototype", page_icon="🧠", layout="wide")
st.title("Graph Reasoning Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("What would you like to know?"):
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        steps_container = st.expander("Reasoning Steps", expanded=True)
        with steps_container:
            steps_placeholder = st.empty()

        recent_steps = deque(maxlen=4)  # keep only last 5
        final_answer_container = st.empty()
        time_container = st.empty()

        entities = extract_entities_from_question(question)
        all_nodes, all_scores = [], []
        for entity in entities:
            nodes, scores = graph.search_nodes(entity, k=3)
            all_nodes.extend(nodes)
            all_scores.extend(scores)

        starting_nodes = filter_relevant_nodes(question, all_nodes, graph)
        agent_answer_nodes: set[Node] = set()
        start_time = time.time()

        for starting_node in starting_nodes:
            subgraph = graph.get_khop_subgraph(starting_node, k=2)
            agent = SubgraphExplorerAgent(
                node=starting_node,
                graph=subgraph,
                question=question,
            )

            for selected_tool, final_answer in agent.answer(max_steps=15):
                tool_name = selected_tool.function.name
                tool_arguments = json.loads(selected_tool.function.arguments)

                # build lines for this step
                lines = []
                if tool_name == "search_in_surroundings":
                    search = tool_arguments.get("query") or tool_arguments.get("type")
                    phrase = random.choice(SEARCH_PHRASES)
                    if search:
                        lines.append(f"{phrase} `{search}` nodes around **{starting_node.name}**")
                    else:
                        lines.append(f"{phrase} nodes around **{starting_node.name}**")

                elif tool_name == "find_paths":
                    source = starting_node.name
                    target = graph.get_node_by_index(tool_arguments.get("dst_node_index")).name
                    phrase = random.choice(CONNECTION_PHRASES)
                    lines.append(f"{phrase} **{source}** and **{target}**")

                elif tool_name == "add_to_answer":
                    idxs = tool_arguments.get("answer_node_indices", [])
                    if idxs:
                        relevant_nodes = [graph.get_node_by_index(i) for i in idxs]
                        lines.append(
                            f"**Found {len(relevant_nodes)} relevant nodes near {starting_node.name}**"
                        )
                        agent_answer_nodes = agent_answer_nodes.union(set(relevant_nodes))

                # add divider after step
                lines.append("---")

                # store only last 5
                recent_steps.append(lines)

                # re-render last 5
                with steps_placeholder.container():
                    for step_lines in recent_steps:
                        for s in step_lines:
                            st.markdown(s)
        
        # Show loading message
        final_answer_container.markdown("### Final Answer")
        final_answer_container.markdown("_Generating KG-grounded answer..._")

        # Generate the answer
        answer = answer_based_on_nodes(
            question=question,
            nodes=list(agent_answer_nodes),
        )

        total_thinking_time = time.time() - start_time

        # Replace with actual answer
        final_answer_container.markdown("### Final Answer")
        final_answer_container.markdown(answer)

        time_container.markdown(f"**Total thinking time: {total_thinking_time:.2f} seconds**")

        full_response = f"{answer}\n\n**Total thinking time: {total_thinking_time:.2f} seconds**"
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        if st.button("Ask something else"):
            st.session_state.messages = []
            st.rerun()