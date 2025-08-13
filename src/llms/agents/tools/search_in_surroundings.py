import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from src.llms.agents.tools.tools import Tool
from graph_types.graph import Graph, Node
from fuzzywuzzy import fuzz


def fuzzy_match(name, pattern, threshold=90):
    if not name:
        return False
    return fuzz.partial_ratio(name.lower(), pattern.lower()) >= threshold


class SearchInSurroundingsTool(Tool):
    MAX_RESULTS_SHOWN: int = 15

    def __init__(self, graph):
        super().__init__(
            name="search_in_surroundings",
            description="Search in surroundings (1 or 2 hop) for a specific keyword, in nodes with a specific type",
            graph=graph,
        )
        self.MAX_RESULTS_SHOWN = 20

    def _format_response(self, node, query, type, k, candidates: list[str]) -> str:
        if not candidates:
            return f"search_in_surroundings(query={query}, type={type}, k={k}) didn't find anything. Consider widening your search or changing the strategy.\n"

        response = ""
        # If asking for the whole hop
        if not query and not type:
            response += (
                f"search_in_surroundings(k={k}) found:\n\n"
                + "\n\n".join(candidates[: self.MAX_RESULTS_SHOWN])
                + "\n\n"
            )
            response += (
                f"(Showing {self.MAX_RESULTS_SHOWN} of {len(candidates)} results. You may want to refine your search)"
                if len(candidates) > self.MAX_RESULTS_SHOWN
                else "" + "\n\n"
            )

        # If filtered by type only
        if type and not query:
            response += (
                f"search_in_surroundings(type={type}, k={k}) found:\n\n"
                + "\n\n".join(candidates[: self.MAX_RESULTS_SHOWN])
                + "\n\n"
            )
            response += (
                f"(Showing {self.MAX_RESULTS_SHOWN} of {len(candidates)} results. You may want to refine your search)"
                if len(candidates) > self.MAX_RESULTS_SHOWN
                else "" + "\n\n"
            )

        if query:
            # If filtered by type and query
            if type:
                response += f"search_in_surroundings(query={query}, type={type}, k={k}) found:\n\n"
            # If filtered by query only
            else:
                response += f"search_in_surroundings(query={query}, k={k}) found:\n\n"

            response += "\n\n".join(candidates[: self.MAX_RESULTS_SHOWN]) + "\n\n"
            response += (
                f"(Showing {self.MAX_RESULTS_SHOWN} of {len(candidates)} results. You may want to refine your search)\n"
                if len(candidates) > self.MAX_RESULTS_SHOWN
                else "" + "\n\n"
            )

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

        # Maybe we should filter candidates here to make this faster. Anyway we are only showing the first 15
        if not query:
            candidates = [
                str(self.graph.get_node_by_index(idx)) for idx in search_nodes_df["index"].tolist()
            ]
            return self._format_response(node, query, type, k, candidates)
        else:
            nodes_matching_name_df = search_nodes_df[
                (search_nodes_df["name"].apply(lambda x: fuzzy_match(x, query)))
                | (search_nodes_df["name"].str.contains(query, case=False, regex=False))
            ]

            candidates: list[str] = (
                [
                    str(self.graph.get_node_by_index(idx))
                    for idx in nodes_matching_name_df["index"].tolist()
                ]
                if nodes_matching_name_df is not None
                else []
            )

            query_words = [word for word in query.split() if len(word) > 3]
            nodes_matching_summary_df = search_nodes_df[
                (~search_nodes_df["index"].isin(nodes_matching_name_df["index"]))
            ]
            if len(nodes_matching_summary_df) > 50_000:
                print()
            nodes_matching_summary_df["appearing_words"] = nodes_matching_summary_df[
                "summary"
            ].apply(
                lambda summary: [word for word in query_words if word.lower() in summary.lower()]
            )
            nodes_matching_summary_df["relevance"] = nodes_matching_summary_df[
                "appearing_words"
            ].apply(len)
            nodes_matching_summary_df = nodes_matching_summary_df[
                nodes_matching_summary_df["relevance"] > 0
            ].sort_values(by="relevance", ascending=False)

            for _, row in nodes_matching_summary_df.iterrows():
                idx = row["index"]
                summary = row["summary"]
                appearing_words = row["appearing_words"]

                candidate = self.graph.get_node_by_index(idx)

                # Find context around matched words
                matched_contexts = []
                for word in appearing_words:
                    word_index = summary.lower().find(word.lower())
                    if word_index != -1:
                        context_start = max(0, word_index - 30)
                        context_end = min(len(summary), word_index + len(word) + 30)
                        context = summary[context_start:context_end]
                        # Highlight the matched word
                        highlighted_context = context.replace(
                            summary[word_index : word_index + len(word)],
                            f"**{summary[word_index:word_index + len(word)]}**",
                        )
                        matched_contexts.append(highlighted_context)

                # Join contexts or use the first one if multiple
                if matched_contexts:
                    context_display = (
                        f"{len(matched_contexts)} matches: {', '.join(matched_contexts[:2])}"
                    )
                    candidates.append(f"{str(candidate)}: {context_display}")
                else:
                    candidates.append(str(candidate))

            return self._format_response(node, query, type, k, candidates)

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


if __name__ == "__main__":
    from graph_types.graph import Graph

    graph = Graph.load("prime")
    tool = SearchInSurroundingsTool(graph)

    node = graph.get_node_by_index(14323)  # A paper node
    print(tool(node, "herpesvirus", "", 2))
