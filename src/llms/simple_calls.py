import json
import os
import pickle

import pandas as pd
from litellm import completion

from graph_types.graph import Graph, Node

if not os.path.exists("data/cache/llm_calls_cache.pkl"):
    LLM_CALLS_CACHE = {}
    os.makedirs("data/cache", exist_ok=True)
    with open("data/cache/llm_calls_cache.pkl", "wb") as f:
        pickle.dump(LLM_CALLS_CACHE, f)


def simple_completion(system_prompt: str, user_prompt: str, use_cache=True) -> str:
    with open("data/cache/llm_calls_cache.pkl", "rb") as f:
        LLM_CALLS_CACHE = pickle.load(f)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    cache_key = f"{system_prompt.strip()}_{user_prompt.strip()}"

    if use_cache and LLM_CALLS_CACHE.get(cache_key):
        print("Using cached llm response")
        return LLM_CALLS_CACHE[cache_key]

    response = (
        completion(
            model="azure/gpt-4o-1120",
            messages=messages,
            temperature=0.1,
            max_tokens=500,
        )
        .choices[0]
        .message.content.strip()
    )

    LLM_CALLS_CACHE[cache_key] = response
    with open("data/cache/llm_calls_cache.pkl", "wb") as f:
        pickle.dump(LLM_CALLS_CACHE, f)

    return response


def parse_response(response: str) -> list[str]:
    try:
        entities = json.loads(response)
    except json.JSONDecodeError:
        # Try to handle common case where response is a Python list string (single quotes)
        try:
            entities = json.loads(response.replace("'", '"'))
        except Exception:
            try:
                entities = response.replace("[", "").replace("]", "").split(", ")
            except Exception:
                print(f"Failed to parse entities from response: {response}")
                entities = []

    return entities


def extract_entities_from_question(question: str) -> list[str]:
    system_prompt = """
    You need to extract the most specific entities from the question.
    Return only the entities as a list, no other text.
    """
    user_prompt = f"Extract entities from this question: {question}"

    response = simple_completion(system_prompt=system_prompt, user_prompt=user_prompt)
    entities = parse_response(response)
    return entities


def extract_question_answer_type(question: str, node_types: list[str]) -> str:
    system_prompt = f"""
    Extract the answer type from the question based on the node types.
    This is to discard the answers that are not semantically relevant to the question.
    The possible node types are: {node_types}.
    Return only the answer type as a string, no other text.
    """
    user_prompt = f"Extract the answer type from this question: {question}"
    answer_type = simple_completion(system_prompt=system_prompt, user_prompt=user_prompt)
    return answer_type


def filter_relevant_nodes(question: str, nodes: list[Node], graph: Graph) -> list[Node]:
    system_prompt = f"""
    You will be given a question about a graph and a list of nodes from that graph.
    The final task will be to find nodes in the graph that respond to the question. For this, we want to identify smaller subgraphs that might be very relevant to the question.
    Your task is to group the nodes into two categories: those that are relevant to the question and those that are not.
    If a node is marked as relevant, an agent will explore it in the next step. If a node is marked as not relevant, it will be discarded.
    
    A node is relevant if it is mentioned in the question or if it represents a concept that is directly related to the question.
    **Very general nodes are usually not relevant to the question, so they should be discarded.**
    
    You must respond with valid JSON only. Return an array of tuples (name, index) for relevant nodes:
    [{{"name": "node_name", "index": node_index}}, {{"name": "node_name", "index": node_index}}, ...]
    """
    user_prompt = f"Question: {question}\nNodes:{[str(n) for n in nodes]}\nReturn only valid JSON array of objects with 'name' and 'index' fields for relevant nodes."
    
    response = simple_completion(
        system_prompt=system_prompt, user_prompt=user_prompt, use_cache=False
    )
    
    node_data = json.loads(response)
    nodes = [graph.get_node_by_index(int(item["index"])) for item in node_data]
    return nodes


# NOTE: I think that asking for the index of the node is not the best way to do this.
def select_starting_node(question: str, sorted_central_nodes: list[Node]) -> Node:
    system_prompt = f"""
    We are working on a graph rag task.
    The idea is to, given a question, select a small subgraph of the graph that is relevant to the question.
    For this, we will select a starting node.
    You will be given a list of nodes sorted by lexical/semantic similarity, and your task is to select the node to start the search from.
    It's likely that the starting node is the first node in the list, but this is not always the case.
    Select the node that is the most explicitly mentioned in the question, if there is more than one, select the most specific one.
    Return the name of the selected node in the list, nothing else
    """

    candidate_nodes = ",\n".join([node.name for node in sorted_central_nodes])
    user_prompt = f"""
    Question: {question}
    Sorted nodes: {candidate_nodes}
    Select the starting node from the sorted nodes.
    """

    name = simple_completion(system_prompt=system_prompt, user_prompt=user_prompt)
    matching_nodes = [node for node in sorted_central_nodes if node.name.lower() == name.lower()]
    if not matching_nodes:
        print(f"WARNING: No matching node found for name: {name}. Returning the first node.")
        return sorted_central_nodes[0]
    return matching_nodes[0]


if __name__ == "__main__":
    # Example usage
    question = "What are the symptoms of diabetes?"
    entities = extract_question_answer_type(question, ["drug", "disease", "symptom"])
    print(f"Extracted entities: {entities}")
