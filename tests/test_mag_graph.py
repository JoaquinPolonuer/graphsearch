import pandas as pd
from src.graph_types.mag import MagGraph

class TestMagGraph:
    def test_load_graph_and_get_node_by_index(self):
        nodes_df = pd.read_csv("data/01_csv_graphs/mag/nodes.csv")
        edges_df = pd.read_csv("data/01_csv_graphs/mag/edges.csv")
        graph = MagGraph(name="mag", nodes_df=nodes_df, edges_df=edges_df)
        graph.get_node_by_index(0)
