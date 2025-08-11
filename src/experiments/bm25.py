import pandas as pd
import sys
from pathlib import Path
from rank_bm25 import BM25Okapi

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import iterate_qas, load_embeddings, load_graph_and_qas, setup_results_dir
from src.experiments.utils import semantic_sort, map_entities_to_nodes, save_log, send_explorers

graph_name = "amazon"
graph, qas = load_graph_and_qas(graph_name)

corpus = graph.nodes_df["summary"].astype(str).tolist()
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)  # slow

results_dir = setup_results_dir(graph.name, "bm25")
for question_index, question, answer_indices in iterate_qas(qas, limit=1000):
    tokenized_query = question.lower().split()

    scores = bm25.get_scores(tokenized_query)  # slow
    bm25_df = pd.DataFrame(
        {"query": question, "bm25_score": scores, "name": corpus, "index": graph.nodes_df.index}
    )
    bm25_df = bm25_df.sort_values(by="bm25_score", ascending=False).reset_index(drop=True)
    bm25_answer_indices = bm25_df["index"].tolist()

    save_log(
        {
            "question": question,
            "answer_indices": answer_indices,
            "bm25_answer_indices": bm25_answer_indices[:100],
        },
        results_dir=results_dir,
        question_index=question_index,
    )
