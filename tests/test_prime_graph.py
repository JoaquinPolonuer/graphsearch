import pandas as pd

from src.graph_types.prime import PrimeGraph, PrimeNode


class TestPrimeGraph:
    def test_load_graph_and_get_node_by_index(self):
        graph = PrimeGraph.load()
        node = graph.get_node_by_index(0)
        assert isinstance(node, PrimeNode)
        assert node.name == "PHYHIP"

    def test_find_node_by_name(self):
        graph = PrimeGraph.load()
        
        IL27_node = graph.get_node_by_index(34082)
        nodes = graph.search_nodes("IL27")
        assert IL27_node in nodes