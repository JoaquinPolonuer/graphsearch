import json

from litellm import completion


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
    return entities
