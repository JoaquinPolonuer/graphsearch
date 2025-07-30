import pandas as pd

from src.graph_types.prime import PrimeGraph


class TestPrimeGraph:
    def test_load_graph_and_get_node_by_index(self):
        nodes_df = pd.read_csv("data/01_csv_graphs/prime/nodes.csv")
        edges_df = pd.read_csv("data/01_csv_graphs/prime/edges.csv")
        graph = PrimeGraph(name="prime", nodes_df=nodes_df, edges_df=edges_df)
        graph.get_node_by_index(0)
