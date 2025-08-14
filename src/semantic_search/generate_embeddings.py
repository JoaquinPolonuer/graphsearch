import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import os
import torch
import pandas as pd
from src.utils import (
    iterate_qas,
    load_graph_and_qas,
    count_tokens,
    truncate_to_token_limit,
)
from openai import AzureOpenAI, OpenAI
import argparse

parser = argparse.ArgumentParser(description="Generate embeddings for a graph.")
parser.add_argument("--graph_name", type=str, required=True, help="Name of the graph to process")
args = parser.parse_args()

graph_name = args.graph_name
graph, qas = load_graph_and_qas(graph_name)

AZURE_ENDPOINT = os.environ["AZURE_API_BASE"]
AZURE_API_KEY = os.environ["AZURE_API_KEY"]
AZURE_API_VERSION = os.environ["AZURE_API_VERSION"]

client = AzureOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_API_KEY:
    fallback_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    fallback_client = None 

EMBEDDINGS_MODEL = "text-embedding-3-small"
EMBEDDINGS_DIR = f"data/graphs/embeddings/{EMBEDDINGS_MODEL}/{graph_name}"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

## Node embeddings
if os.path.exists(f"{EMBEDDINGS_DIR}/node_embeddings.pt"):
    node_embeddings = torch.load(f"{EMBEDDINGS_DIR}/node_embeddings.pt")
else:
    node_embeddings = torch.tensor([])


for i in range(0, len(graph.nodes_df["summary"].tolist()), 100):
    if i + 100 < len(node_embeddings):
        continue

    batch = graph.nodes_df["summary"].tolist()[i : i + 100]

    for j in range(len(batch)):
        summary = batch[j]
        token_count = count_tokens(summary, EMBEDDINGS_MODEL)
        if token_count > 8192:
            batch[j] = truncate_to_token_limit(summary, 8192, EMBEDDINGS_MODEL)
            print(
                f"Truncated summary at index {i+j} from {token_count} to {count_tokens(batch[j])} tokens.", 
                flush=True
            )
    try:
        response = client.embeddings.create(input=batch, model=EMBEDDINGS_MODEL)
    except Exception as e:
        if fallback_client:
            print(f"Azure client failed for batch {i}, trying fallback OpenAI client: {e}")
            try:
                response = fallback_client.embeddings.create(input=batch, model=EMBEDDINGS_MODEL)
            except Exception as fallback_e:
                raise RuntimeError(f"Both Azure and OpenAI clients failed for batch starting at index {i}. Azure error: {e}. OpenAI error: {fallback_e}")
        else:
            raise RuntimeError(f"Error generating embeddings for batch starting at index {i}: {e}")

    batch_embeddings = torch.tensor([data.embedding for data in response.data])
    node_embeddings = torch.cat((node_embeddings, batch_embeddings), dim=0)

    print(
        f"Processed {i + len(batch)} / {len(graph.nodes_df['summary'].tolist())} {graph_name} summaries", 
        flush=True
    )

    if (i + len(batch)) % 1000 == 0:
        torch.save(
            node_embeddings,
            f"{EMBEDDINGS_DIR}/node_embeddings.pt",
        )
        print(f"Saved intermediate node embeddings at {i + len(batch)} nodes.", flush=True)

torch.save(
    node_embeddings,
    f"{EMBEDDINGS_DIR}/node_embeddings.pt",
)

## Question embeddings
questions = (
    pd.DataFrame(iterate_qas(qas), columns=["question_id", "question", "answer_indices"])
    .sort_values("question_id")
    .reset_index(drop=True)
)

if os.path.exists(f"{EMBEDDINGS_DIR}/question_embeddings.pt"):
    question_embeddings = torch.load(f"{EMBEDDINGS_DIR}/question_embeddings.pt")
else:
    question_embeddings = torch.tensor([])

for i in range(0, len(questions), 100):
    if i + 100 < len(question_embeddings):
        continue

    batch = questions["question"].tolist()[i : i + 100]
    try:
        response = client.embeddings.create(input=batch, model=EMBEDDINGS_MODEL)
    except Exception as e:
        if fallback_client:
            print(f"Azure client failed for question batch {i}, trying fallback OpenAI client: {e}")
            try:
                response = fallback_client.embeddings.create(input=batch, model=EMBEDDINGS_MODEL)
            except Exception as fallback_e:
                raise RuntimeError(f"Both Azure and OpenAI clients failed for question batch starting at index {i}. Azure error: {e}. OpenAI error: {fallback_e}")
        else:
            raise RuntimeError(f"Error generating embeddings for question batch starting at index {i}: {e}")

    batch_embeddings = torch.tensor([data.embedding for data in response.data])
    question_embeddings = torch.cat((question_embeddings, batch_embeddings), dim=0)

    print(f"Processed {i + len(batch)} / {len(questions)} {graph_name} questions", flush=True)

torch.save(
    question_embeddings,
    f"{EMBEDDINGS_DIR}/question_embeddings.pt",
)
