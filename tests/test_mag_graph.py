import pandas as pd
import pytest

from graph_types.mag import AuthorNode, FieldOfStudyNode, InstitutionNode, MagGraph, PaperNode


@pytest.fixture
def mag_graph():
    return MagGraph.load()


class TestMagGraph:
    def test_load_graph_and_get_node_by_index(self, mag_graph):
        print(
            f"Loaded graph: {mag_graph.name} with {len(mag_graph.nodes_df)} nodes and {len(mag_graph.edges_df)} edges"
        )

        author = mag_graph.get_node_by_index(0)
        assert isinstance(author, AuthorNode)
        assert author.name == "Alexander Wagenpfahl"

        paper = mag_graph.get_node_by_index(1172724)
        assert isinstance(paper, PaperNode)
        assert (
            paper.name
            == "The Impact of Refractory Material Properties on the Helium Cooled Divertor Design"
        )

        institution = mag_graph.get_node_by_index(1104554)
        assert isinstance(institution, InstitutionNode)
        assert institution.name == "Sangji University"

        field_of_study = mag_graph.get_node_by_index(1113255)
        assert isinstance(field_of_study, FieldOfStudyNode)
        assert field_of_study.name == "Sign function"

    def test_find_node_by_name(self, mag_graph):
        alexander_node = mag_graph.get_node_by_index(0)
        nodes = mag_graph.search_nodes("Alexander Wagenpfahl")
        assert alexander_node in nodes

    def test_get_neighbors(self, mag_graph):
        indian_maritime_node = mag_graph.get_node_by_index(1106065)
        indian_maritime_neighbors = mag_graph.get_neighbors(indian_maritime_node)
        author_from_indian_maritime = mag_graph.get_node_by_index(835790)

        assert isinstance(indian_maritime_neighbors, set)
        assert len(indian_maritime_neighbors) > 0
        assert author_from_indian_maritime in indian_maritime_neighbors
