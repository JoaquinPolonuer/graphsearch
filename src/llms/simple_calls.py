import json
import os
import pickle

from litellm import completion

from graph_types.graph import Graph, Node
from src.prompts.prompts import (
    ENTITY_EXTRACTION_SYSTEM,
    MAG_STARTING_NODE_FILTERING_SYSTEM,
    QUESTION_ANSWER_TYPE_SYSTEM,
    STARTING_NODE_FILTERING_SYSTEM,
)

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
        .message.content
    )
        
    try:
        response = response.strip()
    except Exception as e:
        print(f"Error processing response: {e}")
        response = "[]"
        pass

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
    user_prompt = f"Extract entities from this question: {question}"

    response = simple_completion(system_prompt=ENTITY_EXTRACTION_SYSTEM, user_prompt=user_prompt)
    entities = parse_response(response)
    return entities


def extract_question_answer_type(question: str, node_types: list[str]) -> str:
    user_prompt = f"Extract the answer type from this question: {question}"
    answer_type = simple_completion(
        system_prompt=QUESTION_ANSWER_TYPE_SYSTEM.format(node_types=", ".join(node_types)),
        user_prompt=user_prompt,
    )
    return answer_type


def filter_relevant_nodes(question: str, nodes: list[Node], graph: Graph) -> list[Node]:
    user_prompt = f"Question: {question}\nNodes:{[str(n) for n in nodes]}"
    system_prompt = (
        MAG_STARTING_NODE_FILTERING_SYSTEM
        if graph.name == "mag"
        else STARTING_NODE_FILTERING_SYSTEM
    )

    response = completion(
        model="azure/gpt-4o-1120",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=500,
    )

    response_content = response.choices[0].message.content
    parsed_response = json.loads(response_content)
    node_data = parsed_response["relevant_nodes"]

    result_nodes = [graph.get_node_by_index(int(item["index"])) for item in node_data]
    return result_nodes


def answer_based_on_nodes(question: str, nodes: list[Node]) -> str:
    system_prompt = """
    You are an expert in biomedical knowledge. You will be given a question and a set of nodes extracted from a KG that contain relevant information. 
    Your task is to provide a concise answer based only on the information in the nodes.
    """

    evidence = ""
    for node in nodes:
        evidence += f"Node Name: {node.name}\nDetails: {node.details}\n\n"

    user_prompt = f"Question: {question}\n\nEvidence:\n\n{evidence}"

    return simple_completion(system_prompt=system_prompt, user_prompt=user_prompt)


if __name__ == "__main__":
    # Example usage
    question = "What are the symptoms of diabetes?"
    entities = extract_question_answer_type(question, ["drug", "disease", "symptom"])
    print(f"Extracted entities: {entities}")
