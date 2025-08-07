from typing import Any
from src.llms.agents.tools.tools import Tool
from graph_types.graph import Graph, Node
from fuzzywuzzy import fuzz


def fuzzy_match(name, pattern, threshold=90):
    return fuzz.partial_ratio(name.lower(), pattern.lower()) >= threshold


class SearchInSurroundingsTool(Tool):
    def __init__(self, agent, graph):
        super().__init__(
            name="search_in_surroundings",
            description="Search in surroundings (1 or 2 hop) for a specific keyword, in nodes with a specific type",
            graph=graph,
        )

    def _format_response(
        self, node, query, type, k, candidates: list[str], candidates_matching_details: list[str]
    ) -> str:
        if not candidates:  # and (not candidates_matching_details):
            return f"search_in_surroundings(query={query}, type={type}, k={k}) didn't find anything. Consider widening your search or changing the strategy.\n"

        response = ""
        # If asking for the whole hop
        if not query and not type:
            response += (
                f"search_in_surroundings(k={k}) found:\n\n" + "\n\n".join(candidates[:15]) + "\n\n"
            )
            response += (
                f"(Showing 15 of {len(candidates)} results. You may want to refine your search)"
                if len(candidates) > 15
                else "" + "\n\n"
            )

        # If filtered by type only
        if type and not query:
            response += (
                f"search_in_surroundings(type={type}, k={k}) found:\n\n"
                + "\n\n".join(candidates[:15])
                + "\n\n"
            )
            response += (
                f"(Showing 15 of {len(candidates)} results. You may want to refine your search)"
                if len(candidates) > 15
                else "" + "\n\n"
            )

        if query:
            # If filtered by type and query
            if type:
                response += f"search_in_surroundings(query={query}, type={type}, k={k}) found:\n\n"
            # If filtered by query only
            else:
                response += f"search_in_surroundings(query={query}, k={k}) found:\n\n"

            # if candidates:
            response += "\n\n".join(candidates[:15]) + "\n\n"
            response += (
                f"(Showing 15 of {len(candidates)} results. You may want to refine your search)\n"
                if len(candidates) > 15
                else "" + "\n\n"
            )
            # else:
            #     response += "\n\n".join(candidates_matching_details[:5]) + "\n\n"
            #     response += (
            #         f"(Showing 5 of {len(candidates_matching_details)} results. You may want to refine your search)\n"
            #         if len(candidates_matching_details) > 5
            #         else "" + "\n\n"
            #     )

        response += (
            "If you found what you were looking for you can submit answer.\n"
            f"You can also find connections between the current node `{node.name}` and any of the candidate nodes by using the `find_paths` function.\n"
        )

        return response

    def __call__(self, node: Node, query: str, type: str, k: int) -> str:
        if not k in [1, 2]:
            return f"search_in_surroundings({query}, {type}, {k}) only supports k=1 or k=2.\n"

        search_nodes_df = self.graph.get_khop_subgraph(node, k).nodes_df

        if type:
            search_nodes_df = search_nodes_df[search_nodes_df["type"] == type]

        if not query:
            candidates = [
                str(self.graph.get_node_by_index(idx)) for idx in search_nodes_df["index"].tolist()
            ]
            return self._format_response(node, query, type, k, candidates, [])
        else:
            nodes_matching_name_df = search_nodes_df[
                (search_nodes_df["name"].apply(lambda x: fuzzy_match(x, query)))
                | (search_nodes_df["name"].str.contains(query, case=False))
            ]

            # nodes_matching_details_df = search_nodes_df[
            #     (search_nodes_df["details"].str.contains(query, case=False))
            # ]

            # candidates_matching_details: list[str] = []
            # for idx in nodes_matching_details_df["index"].tolist():
            #     candidate = self.graph.get_node_by_index(idx)

            #     match_in_candidate = ""
            #     for sentence in str(candidate.details).split("."):
            #         if query.lower() in sentence.lower():
            #             match_in_candidate += f"(...) {sentence}"

            #     candidates_matching_details.append(f"{str(candidate)}: {match_in_candidate}")

            candidates_matching_name: list[str] = (
                [
                    str(self.graph.get_node_by_index(idx))
                    for idx in nodes_matching_name_df["index"].tolist()
                ]
                if nodes_matching_name_df is not None
                else []
            )
            return self._format_response(
                node, query, type, k, candidates_matching_name, []  # candidates_matching_details
            )

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The keyword to search for. You can also leave this empty and get all the `type` nodes",
                        },
                        "type": {
                            "type": "string",
                            "description": "The type of nodes to search in",
                        },
                        "k": {
                            "type": "integer",
                            "description": "The number of hops (1 or 2) to search within",
                        },
                    },
                    "required": ["k"],
                },
            },
        }
