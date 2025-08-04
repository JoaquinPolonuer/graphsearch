import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd

from config import DATA_DIR
from graph_types.mag import MagGraph
from llms.simple_calls import extract_entities_from_question

qas = pd.read_csv(DATA_DIR / "02_qa_datasets/mag.csv")
mag_graph = MagGraph.load()
doc_embeddings = torch.load(
    DATA_DIR / f"node_embeddings/{mag_graph.name}/text-embedding-ada-002/doc/candidate_emb_dict.pt"
)
query_embeddings = torch.load(
    DATA_DIR / f"node_embeddings/{mag_graph.name}/text-embedding-ada-002/query/query_emb_dict.pt"
)
results_dir = DATA_DIR / "connectedness/mag_logs"
os.makedirs(results_dir, exist_ok=True)
for question_index, row in qas.iloc[:100].iterrows():
    question = row["question"]
    question_embedding = query_embeddings[question_index][0]
    answer_indices = json.loads(row["answer_indices"])

    entities = extract_entities_from_question(question)

    central_nodes = []
    for entity in entities:
        nodes = mag_graph.search_nodes(entity, k=1)
        central_nodes.extend(nodes)

    neighborhoods = []
    for node in central_nodes:
        neighbors = mag_graph.get_neighbors(node)
        neighborhoods.append(neighbors)

    candidate_rank = {}
    for neighborhood in neighborhoods:
        for node in neighborhood:
            if node not in candidate_rank:
                candidate_rank[node] = 0
            candidate_rank[node] += 1

    candidate_df = pd.DataFrame(
        [(node.name, node.index, node.type, value) for node, value in candidate_rank.items()],
        columns=["name", "index", "type", "connectedness"],
    )

    candidate_df = candidate_df[candidate_df["type"] == "paper"]

    candidate_df["embedding"] = candidate_df["index"].apply(lambda index: doc_embeddings[index])
    candidate_df["similarity"] = candidate_df["embedding"].apply(
        lambda embedding: torch.matmul(
            question_embedding.detach().clone(), embedding.detach().clone().T
        ).item()
    )

    candidate_df = candidate_df.sort_values(
        ["connectedness", "similarity"], ascending=[False, False]
    )

    retrieved_indices = candidate_df["index"].tolist()
    hit_1 = retrieved_indices[0] in answer_indices if retrieved_indices else False
    hit_5 = any([retrieved_index in answer_indices for retrieved_index in retrieved_indices[:5]])
    recall_20 = len(set(retrieved_indices[:20]) & set(answer_indices)) / len(answer_indices)

    log = {
        "question": question,
        "entities": [entity for entity in entities],
        "central_nodes": [node.to_doc() for node in central_nodes],
        "retrieved_indices": retrieved_indices,
        "answer_indices": answer_indices,
        "hit@1": hit_1,
        "hit@5": hit_5,
        "recall@20": recall_20,
    }

    with open(results_dir / f"{question_index}.json", "w") as f:
        json.dump(log, f, indent=4)
        candidate_df.to_csv(results_dir / f"{question_index}_candidates.csv", index=False)

    print(f"Processed question {question_index}")
