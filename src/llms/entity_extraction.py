import os
import json
import pandas as pd

from litellm import completion
from graph_types.graph import Node

if os.path.exists("data/02_qa_datasets/entity_extraction_cache.csv"):
    QUESTION_ENTITY_CACHE = pd.read_csv("data/02_qa_datasets/entity_extraction_cache.csv")
else:
    QUESTION_ENTITY_CACHE = pd.DataFrame(columns=["question", "entities"])

if not os.path.exists("data/02_qa_datasets/answer_type_cache.csv"):
    ANSWER_TYPE_CACHE = pd.DataFrame(columns=["question", "answer_type"])
else:
    ANSWER_TYPE_CACHE = pd.read_csv("data/02_qa_datasets/answer_type_cache.csv")


def simple_completion(system_prompt: str, user_prompt: str) -> str:

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = (
        completion(model="azure/gpt-4o-1120", messages=messages, temperature=0.1, max_tokens=500)
        .choices[0]
        .message.content.strip()
    )

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

    if question in QUESTION_ENTITY_CACHE["question"].values:
        cached_entities = QUESTION_ENTITY_CACHE[QUESTION_ENTITY_CACHE["question"] == question][
            "entities"
        ].values[0]
        print("Using cached entities for question:", question[:30], "...")
        return json.loads(cached_entities)

    system_prompt = """
    You need to extract the most specific entities from the question.
    Return only the entities as a list, no other text.
    """

    user_prompt = f"Extract entities from this question: {question}"

    response = simple_completion(system_prompt=system_prompt, user_prompt=user_prompt)
    entities = parse_response(response)

    QUESTION_ENTITY_CACHE.loc[len(QUESTION_ENTITY_CACHE)] = [question, json.dumps(entities)]
    QUESTION_ENTITY_CACHE.to_csv("data/02_qa_datasets/entity_extraction_cache.csv", index=False)
    return entities


def extract_question_answer_type(
    question: str, node_types: set[str], model: str = "azure/gpt-4o-1120"
) -> str:
    if question in ANSWER_TYPE_CACHE["question"].values:
        cached_answer_type = ANSWER_TYPE_CACHE[ANSWER_TYPE_CACHE["question"] == question][
            "answer_type"
        ].values[0]
        print("Using cached answer type for question:", question[:30], "...")
        return cached_answer_type

    system_prompt = f"""
    Extract the answer type from the question based on the node types.
    This is to discard the answers that are not semantically relevant to the question.
    The possible node types are: {node_types}.
    Return only the answer type as a string, no other text.
    """

    user_prompt = f"Extract the answer type from this question: {question}"

    answer_type = simple_completion(system_prompt=system_prompt, user_prompt=user_prompt)
    if answer_type not in node_types:
        print(f"Extracted answer type '{answer_type}' is not in the node types: {node_types}.")
        answer_type = "unknown"

    return answer_type


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
    entities = extract_entities_from_question(question)
    print(f"Extracted entities: {entities}")
