import pandas as pd
import pytest

from graph_types.prime import PrimeGraph, PrimeNode


@pytest.fixture
def prime_graph():
    return PrimeGraph.load()


class TestPrimeGraph:
    def test_load_graph_and_get_node_by_index(self, prime_graph):
        node = prime_graph.get_node_by_index(0)
        assert isinstance(node, PrimeNode)
        assert node.name == "PHYHIP"

    def test_find_node_by_name(self, prime_graph):
        IL27_node = prime_graph.get_node_by_index(34082)
        nodes = prime_graph.search_nodes("IL27")
        assert IL27_node in nodes

    def test_get_neighbors(self, prime_graph):
        alzheimer_node = prime_graph.get_node_by_index(28780)
        familial_alzheimer = prime_graph.get_node_by_index(32654)
        alzheimer_neighbors = prime_graph.get_neighbors(alzheimer_node)

        assert isinstance(alzheimer_neighbors, set)
        assert len(alzheimer_neighbors) > 0
        assert familial_alzheimer in alzheimer_neighbors
