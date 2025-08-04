import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from index import ElasticsearchIndex

from graph_types.graph import Graph

graph = Graph.load("amazon")
index = ElasticsearchIndex(name=f"{graph.name}_index")
index.delete_if_exists()
index.create(
    mapping={
        "mappings": {
            "properties": {
                "name": {"type": "text", "analyzer": "standard"},
                "index": {"type": "text"},
                "type": {"type": "keyword"},
            }
        }
    },
)

index.upload_graph(graph)
