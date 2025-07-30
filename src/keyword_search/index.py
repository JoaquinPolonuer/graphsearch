import json
import sys
from pathlib import Path

import requests
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


class ElasticsearchIndex(BaseModel):
    name: str
    base_url: str = "http://localhost:9200"

    def send_batch(self, batch):
        # Convert batch to NDJSON format
        bulk_data = "\n".join([str(item).replace("'", '"') for item in batch]) + "\n"

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
                        {"match": {"name": {"query": query, "boost": 2}}},
                        {"wildcard": {f"{'name'}.keyword": f"*{query}*"}},
                        {"fuzzy": {"name": {"value": query, "fuzziness": "AUTO"}}},
                    ]
                }
            },
            "sort": [{"_score": {"order": "desc"}}],
        }

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
