import os
import json
import pandas as pd
import pickle

from litellm import completion
from graph_types.graph import Node

if not os.path.exists("data/cache/llm_calls_cache.pkl"):
    LLM_CALLS_CACHE = {}
else:
    os.makedirs("data/cache", exist_ok=True)
    with open("data/cache/llm_calls_cache.pkl", "rb") as f:
        LLM_CALLS_CACHE = pickle.load(f)


def simple_completion(system_prompt: str, user_prompt: str) -> str:

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    messages_as_str = str(messages)

    if LLM_CALLS_CACHE.get(messages_as_str):
        print("Using cached llm response")
        return LLM_CALLS_CACHE[messages_as_str]

    response = (
        completion(model="azure/gpt-4o-1120", messages=messages, temperature=0.1, max_tokens=500)
        .choices[0]
        .message.content.strip()
    )

    LLM_CALLS_CACHE[messages_as_str] = response
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

#NOTE: I think that asking for the index of the node is not the best way to do this.
def select_starting_node(question: str, sorted_central_nodes: list[Node]) -> Node:
    system_prompt = f"""
    We are working on a graph rag task.
    The idea is to, given a question, select a small subgraph of the graph that is relevant to the question.
    For this, we will select a starting node.
    You will be given a list of nodes sorted by lexical/semantic similarity, and your task is to select the node to start the search from.
    It's likely that the starting node is the first node in the list, but this is not always the case.
    Select the node that is the most explicitly mentioned in the question, if there is more than one, select the most specific one.
    Return the index of the selected node in the list, nothing else
    """

    candidate_nodes = ",\n".join([node.name for node in sorted_central_nodes])
    user_prompt = f"""
    Question: {question}
    Sorted nodes: {candidate_nodes}
    Select the starting node from the sorted nodes.
    """

    i = int(simple_completion(system_prompt=system_prompt, user_prompt=user_prompt))
    return sorted_central_nodes[i]


if __name__ == "__main__":
    # Example usage
    question = "What are the symptoms of diabetes?"
    entities = extract_question_answer_type(question, ["drug", "disease", "symptom"])
    print(f"Extracted entities: {entities}")
