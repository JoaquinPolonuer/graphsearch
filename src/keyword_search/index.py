import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import tantivy
from pydantic import BaseModel

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
import time

from graph_types.graph import Graph
from config import DATA_DIR


class TantivyIndex(BaseModel):
    name: str
    index_path: Path = DATA_DIR / "graphs" / "indices"

    def __init__(self, **data):
        super().__init__(**data)
        self._index = None
        self._schema = None
        self._query_parser = None

    @property
    def full_path(self) -> Path:
        return self.index_path / self.name

    def _create_schema(self):
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("name", stored=True)
        schema_builder.add_integer_field("index", stored=True)
        schema_builder.add_text_field("type", stored=True)
        schema_builder.add_text_field("summary", stored=True)
        return schema_builder.build()

    def delete_if_exists(self):
        if self.full_path.exists():
            print(f"Index {self.name} exists. Deleting...")
            import shutil
            shutil.rmtree(self.full_path)
            print(f"Index {self.name} deleted successfully")
        else:
            print(f"Index {self.name} does not exist")
        return True

    def create(self, mapping=None):
        print(f"Creating index {self.name}...")
        
        # Create directory if it doesn't exist
        self.full_path.mkdir(parents=True, exist_ok=True)
        
        # Create schema
        self._schema = self._create_schema()
        
        # Create index
        self._index = tantivy.Index(self._schema, path=str(self.full_path))
        
        print(f"Index {self.name} created successfully")
        return True

    def upload_graph(self, graph: Graph, batch_size: int = 1000):
        if not self._index:
            self._load_index()
        
        total_docs = 0
        writer = self._index.writer()
        
        for idx in range(len(graph.nodes_df)):
            node = graph.get_node_by_index(idx)
            doc_data = node.to_doc()
            
            # Create Tantivy document
            doc = tantivy.Document()
            doc.add_text("name", doc_data["name"])
            doc.add_integer("index", doc_data["index"])
            doc.add_text("type", doc_data["type"])
            doc.add_text("summary", doc_data["summary"])
            
            writer.add_document(doc)
            total_docs += 1
            
            if total_docs % batch_size == 0:
                print(f"Indexed {total_docs} documents...", flush=True)
        
        # Commit all documents
        writer.commit()
        print("Import completed!", flush=True)
        print(f"Total documents in {self.name}: {total_docs}", flush=True)

    def _load_index(self):
        if not self._index and self.full_path.exists():
            self._schema = self._create_schema()
            self._index = tantivy.Index.open(str(self.full_path))

    def search(self, query: str, k: int = 10) -> dict:
        if not self._index:
            self._load_index()
        
        if not self._index:
            return {"hits": {"hits": [], "total": {"value": 0}}}
        
        # Parse query - use the index's parse_query method
        parsed_query = self._index.parse_query(query, ["name", "summary"])
        
        # Search
        searcher = self._index.searcher()
        search_result = searcher.search(parsed_query, k)
        
        # Convert results to Elasticsearch-compatible format
        hits = []
        for score, doc_address in search_result.hits:
            doc = searcher.doc(doc_address)
            source = {
                "name": doc.get_first("name"),
                "index": doc.get_first("index"),
                "type": doc.get_first("type"),
                "summary": doc.get_first("summary") or "",
            }
            hits.append({
                "_source": source,
                "_score": score
            })
        
        return {
            "hits": {
                "hits": hits,
                "total": {"value": len(hits)}
            }
        }

    def search_summary(self, query: str, k: int = 10) -> dict:
        if not self._index:
            self._load_index()
        
        if not self._index:
            return {"hits": {"hits": [], "total": {"value": 0}}}
        
        # Parse query - search only in summary field
        parsed_query = self._index.parse_query(query, ["summary"])
        
        # Search
        searcher = self._index.searcher()
        search_result = searcher.search(parsed_query, k)
        
        # Convert results to Elasticsearch-compatible format
        hits = []
        for score, doc_address in search_result.hits:
            doc = searcher.doc(doc_address)
            source = {
                "name": doc.get_first("name"),
                "index": doc.get_first("index"),
                "type": doc.get_first("type"),
                "summary": doc.get_first("summary") or "",
            }
            hits.append({
                "_source": source,
                "_score": score
            })
        
        return {
            "hits": {
                "hits": hits,
                "total": {"value": len(hits)}
            }
        }

if __name__ == "__main__":
    # Get all existing indices
    indices_dir = DATA_DIR / "graphs" / "indices"
    
    if not indices_dir.exists():
        print("No indices directory found")
        exit()
    
    index_dirs = [d for d in indices_dir.iterdir() if d.is_dir()]
    
    if not index_dirs:
        print("No indices found")
        exit()
    
    print("=== INDEX SUMMARY ===")
    indices_info = []
    
    for index_dir in index_dirs:
        index_name = index_dir.name
        try:
            index = TantivyIndex(name=index_name)
            index._load_index()
            
            if index._index:
                # Get document count
                searcher = index._index.searcher()
                doc_count = searcher.num_docs
                indices_info.append((index_name, doc_count, index))
                print(f"{index_name}: {doc_count} documents")
            else:
                print(f"{index_name}: Failed to load")
        except Exception as e:
            print(f"{index_name}: Error - {e}")
    
    print(f"\nTotal indices: {len(indices_info)}")
    
    # Sample search on each index
    print("\n=== SAMPLE SEARCHES ===")
    search_query = "epithelial skin neoplasms"
    
    for index_name, doc_count, index in indices_info:
        print(f"\n--- {index_name} ({doc_count} docs) ---")
        
        try:
            results = index.search(search_query, k=1)
            
            if results['hits']['hits']:
                hit = results['hits']['hits'][0]
                source = hit['_source']
                score = hit['_score']
                
                print(f"Score: {score:.3f}")
                print(f"Name: {source.get('name', 'N/A')}")
                print(f"Type: {source.get('type', 'N/A')}")
                summary = source.get('summary', 'N/A')
                print(f"Summary: {summary[:100]}{'...' if len(summary) > 100 else ''}")
            else:
                print("No results found")
        except Exception as e:
            print(f"Search error: {e}")
    
    print("\n=== DONE ===")