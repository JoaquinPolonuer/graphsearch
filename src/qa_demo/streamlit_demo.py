import streamlit as st
import time
import random

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

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if question := st.chat_input("What would you like to know?"):
    # Display user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Display assistant response
    with st.chat_message("assistant"):
        steps_container = st.expander("Reasoning Steps", expanded=True)
        final_answer_container = st.empty()
        time_container = st.empty()

        # Extract entities and get starting nodes
        entities = extract_entities_from_question(question)
        all_nodes = []
        all_scores = []
        for entity in entities:
            nodes, scores = graph.search_nodes(entity, k=3)
            all_nodes.extend(nodes)
            all_scores.extend(scores)

        starting_nodes = filter_relevant_nodes(question, all_nodes, graph)

        # Initialize agent processing
        agent_answer_nodes: set[Node] = set()
        start_time = time.time()
        step_count = 0

        # Process each starting node with real-time display
        for starting_node in starting_nodes:
            subgraph = graph.get_khop_subgraph(starting_node, k=2)
            agent = SubgraphExplorerAgent(
                node=starting_node,
                graph=subgraph,
                question=question,
            )

            # Stream reasoning steps in real-time
            for selected_tool, final_answer in agent.answer(max_steps=15):
                step_count += 1
                tool_name = selected_tool.function.name
                tool_arguments = json.loads(selected_tool.function.arguments)

                # Get the last tool response from message history
                # tool_response = ""
                # if len(agent.message_history) >= 2:
                #     last_message = agent.message_history[-1]
                #     if last_message.get("role") == "tool":
                #         tool_response = last_message.get("content", "")

                # Display step in real-time
                with steps_container:

                    if tool_name == "search_in_surroundings":
                        search = (
                            tool_arguments.get("query")
                            if tool_arguments.get("query")
                            else tool_arguments.get("type")
                        )

                        search_phrase = random.choice(SEARCH_PHRASES)
                        if search:
                            st.markdown(
                                f"{search_phrase} `{search}` nodes around **{starting_node.name}**"
                            )
                        else:
                            st.markdown(f"{search_phrase} nodes around **{starting_node.name}**")
                        # st.markdown(tool_response)

                    if tool_name == "find_paths":
                        source = starting_node.name
                        target = graph.get_node_by_index(tool_arguments.get("dst_node_index")).name
                        connection_phrase = random.choice(CONNECTION_PHRASES)
                        st.markdown(f"{connection_phrase} **{source}** and **{target}**")
                        # st.markdown(tool_response)

                    if tool_name == "submit_answers":
                        relevant_nodes = tool_arguments.get("answer_node_indices", [])
                        if relevant_nodes:
                            st.markdown(
                                f"**Found {len(relevant_nodes)} relevant nodes near {starting_node.name}**"
                            )
                        else:
                            st.markdown(
                                f"*Didn't find relevant information near {starting_node.name}**"
                            )

                    st.divider()

                if final_answer is not None:
                    agent_answer_nodes = agent_answer_nodes.union(set(final_answer))
                    break

        # Generate and display final answer
        answer = answer_based_on_nodes(
            question=question,
            nodes=list(agent_answer_nodes),
        )

        total_thinking_time = time.time() - start_time

        final_answer_container.markdown(f"### Final Answer")
        final_answer_container.markdown(answer)

        time_container.markdown(f"**Total thinking time: {total_thinking_time:.2f} seconds**")

        # Add the final response to chat history
        full_response = f"{answer}\n\n**Total thinking time: {total_thinking_time:.2f} seconds**"
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Add button to ask something else
        if st.button("Ask something else"):
            st.session_state.messages = []
            st.rerun()
