import requests
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.graph_types.prime import PrimeGraph
from keyword_search.index import ElasticsearchIndex

graph = PrimeGraph.load()
index = ElasticsearchIndex(name=f"{graph.name}_index")
index.delete_if_exists()

index.create(
    mapping={
        "mappings": {
            "properties": {
                "name": {"type": "text", "analyzer": "standard"},
                "index": {"type": "text"},
                "type": {"type": "keyword"},
                "source": {"type": "keyword"},
                "details": {"type": "text", "analyzer": "standard"},
            }
        }
    },
)

batch_size = 10000
batch = []
total_docs = 0

for idx in range(len(graph.nodes_df)):
    doc = graph.get_node_by_index(idx).to_doc()

    batch.append({"index": {"_index": index.name, "_id": doc["index"]}})
    batch.append(doc)

    # Send batch when it reaches batch_size
    if len(batch) >= batch_size * 2:  # *2 because each doc needs 2 lines
        index.send_batch(batch)
        total_docs += batch_size
        print(f"Indexed {total_docs} documents...")
        batch = []

# Send remaining documents
if batch:
    index.send_batch(batch)
    total_docs += len(batch) // 2
    print(f"Indexed {total_docs} documents...")

print("Import completed!")

time.sleep(1)
stats = index.stats()
doc_count = stats["indices"][index.name]["total"]["docs"]["count"]
print(f"Total documents in {index.name}: {doc_count}")
