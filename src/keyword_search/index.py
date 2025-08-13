import json
import sys
from pathlib import Path

import requests
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
import time

from graph_types.graph import Graph


class ElasticsearchIndex(BaseModel):
    name: str
    base_url: str = "http://localhost:9200"

    def send_batch(self, batch):
        lines = []
        for i in range(0, len(batch), 2):
            action = json.dumps(batch[i])
            doc = json.dumps(batch[i + 1])
            lines.extend([action, doc])
        bulk_data = "\n".join(lines) + "\n"

        response = requests.post(
            f"{self.base_url}/_bulk",
            data=bulk_data,
            headers={"Content-Type": "application/x-ndjson"},
        )

        if response.status_code not in (200, 201):
            print(f"Error in batch indexing: {response.text}")
        else:
            # Check for individual document errors
            result = response.json()
            if result.get("errors"):
                for item in result.get("items", []):
                    if "index" in item and item["index"].get("error"):
                        print(
                            f"Error indexing document {item['index']['_id']}: {item['index']['error']}"
                        )

    def delete_if_exists(self):
        print(f"Checking if index {self.name} exists...")
        response = requests.head(f"{self.base_url}/{self.name}")
        if response.status_code == 200:
            print(f"Index {self.name} exists. Deleting...")
            delete_response = requests.delete(f"{self.base_url}/{self.name}")
            if delete_response.status_code == 200:
                print(f"Index {self.name} deleted successfully")
            else:
                print(f"Error deleting index: {delete_response.text}")
                return False
        else:
            print(f"Index {self.name} does not exist")

        return True

    def create(self, mapping):
        print(f"Creating index {self.name} with mapping...")

        response = requests.put(
            f"{self.base_url}/{self.name}",
            json=mapping,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code in (200, 201):
            print(f"Index {self.name} created successfully")
            return True
        else:
            print(f"Error creating index: {response.text}")
            return False

    def upload_graph(self, graph: Graph, batch_size: int = 1000):
        batch = []
        total_docs = 0
        for idx in range(len(graph.nodes_df)):
            doc = graph.get_node_by_index(idx).to_doc()

            batch.append({"index": {"_index": self.name, "_id": doc["index"]}})
            batch.append(doc)

            if len(batch) >= batch_size * 2:  # *2 because each doc needs 2 lines
                self.send_batch(batch)
                total_docs += batch_size
                print(f"Indexed {total_docs} documents...", flush=True)
                batch = []

        # Send remaining documents
        if batch:
            self.send_batch(batch)
            total_docs += len(batch) // 2
            print(f"Indexed {total_docs} documents...", flush=True)

        print("Import completed!", flush=True)

        time.sleep(1)
        stats = self.stats()
        doc_count = stats["indices"][self.name]["total"]["docs"]["count"]
        print(f"Total documents in {self.name}: {doc_count}", flush=True)

    def stats(self):
        response = requests.get(f"{self.base_url}/{self.name}/_stats")
        stats = response.json()
        return stats

    def search(self, query: str, k: int = 10) -> dict:
        search_body = {
            "size": k,
            "query": {
                "bool": {
                    "should": [
                        {"match": {"name": {"query": query, "boost": 3}}},
                        {"match_phrase": {"name": {"query": query, "boost": 2}}},
                        {"match": {"name": {"query": query, "fuzziness": "AUTO", "boost": 1}}},
                    ],
                    "minimum_should_match": 1,
                }
            },
        }

        response = requests.post(
            f"{self.base_url}/{self.name}/_search",
            json=search_body,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        return response.json()

    def search_summary(self, query: str, k: int = 10) -> dict:
        search_body = {"size": k, "query": {"match": {"summary": {"query": query}}}}

        response = requests.post(
            f"{self.base_url}/{self.name}/_search",
            json=search_body,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        return response.json()


if __name__ == "__main__":
    # Example usage
    index = ElasticsearchIndex(name="prime_index")
    results = index.search("epithelial skin neoplasms", k=5)

    if results:
        print(f"Found {results['hits']['total']['value']} results:")
        print("-" * 50)

        for hit in results["hits"]["hits"]:
            source = hit["_source"]
            score = hit["_score"]
            print(f"Score: {score}")
            print(f"Name: {source.get('name', 'N/A')}")
            print(f"Type: {source.get('type', 'N/A')}")
            print(f"Details: {source.get('details', 'N/A')[:100]}...")
            print("-" * 50)
