import os
import json
import pandas as pd

from litellm import completion

if os.path.exists("data/02_qa_datasets/entity_extraction_cache.csv"):
    QUESTION_ENTITY_CACHE = pd.read_csv("data/02_qa_datasets/entity_extraction_cache.csv")
else:
    QUESTION_ENTITY_CACHE = pd.DataFrame(columns=["question", "entities"])


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


def extract_entities_from_question(question: str, model: str = "azure/gpt-4o-1120") -> list[str]:

    if question in QUESTION_ENTITY_CACHE["question"].values:
        cached_entities = QUESTION_ENTITY_CACHE[QUESTION_ENTITY_CACHE["question"] == question]["entities"].values[0]
        print("Using cached entities for question:", question[:30], "...")
        return json.loads(cached_entities)

    system_prompt = """
    You need to extract the most specific entities from the question.
    Return only the entities as a list, no other text.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Extract entities from this question: {question}"},
    ]

    response = (
        completion(model=model, messages=messages, temperature=0.1, max_tokens=500)
        .choices[0]
        .message.content.strip()
    )
    entities = parse_response(response)

    QUESTION_ENTITY_CACHE.loc[len(QUESTION_ENTITY_CACHE)] = [question, json.dumps(entities)]
    QUESTION_ENTITY_CACHE.to_csv("data/02_qa_datasets/entity_extraction_cache.csv", index=False)
    return entities

if __name__ == "__main__":
    # Example usage
    question = "What are the symptoms of diabetes?"
    entities = extract_entities_from_question(question)
    print(f"Extracted entities: {entities}")
