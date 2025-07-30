import pandas as pd

from src.graph_types.mag import MagGraph, PaperNode, AuthorNode, InstitutionNode, FieldOfStudyNode


class TestMagGraph:
    def test_load_graph_and_get_node_by_index(self):
        graph = MagGraph.load()
        print(
            f"Loaded graph: {graph.name} with {len(graph.nodes_df)} nodes and {len(graph.edges_df)} edges"
        )

        author = graph.get_node_by_index(0)
        assert isinstance(author, AuthorNode)
        assert author.name == "Alexander Wagenpfahl"
                     
        paper = graph.get_node_by_index(1172724)
        assert isinstance(paper, PaperNode)
        assert paper.name == "The Impact of Refractory Material Properties on the Helium Cooled Divertor Design"
        
        institution = graph.get_node_by_index(1104554)
        assert isinstance(institution, InstitutionNode)
        assert institution.name == "Sangji University"
        
        field_of_study = graph.get_node_by_index(1113255)
        assert isinstance(field_of_study, FieldOfStudyNode)
        assert field_of_study.name == "Sign function"
