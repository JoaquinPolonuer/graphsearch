import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from index import ElasticsearchIndex

from graph_types.graph import Graph
import argparse

parser = argparse.ArgumentParser(description="Generate embeddings for a graph.")
parser.add_argument("--graph_name", type=str, required=True, help="Name of the graph to process")
args = parser.parse_args()

graph_name = args.graph_name

graph = Graph.load(graph_name)
index = ElasticsearchIndex(name=f"{graph.name}_index")
index.delete_if_exists()
index.create(
    mapping={
        "mappings": {
            "properties": {
                "name": {"type": "text", "analyzer": "standard"},
                "index": {"type": "text"},
                "type": {"type": "keyword"},
                "summary": {"type": "text", "analyzer": "standard"},
            }
        }
    },
)

index.upload_graph(graph)
