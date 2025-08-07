from typing import Any
from src.llms.agents.tools.tools import Tool


class FindPathsTool(Tool):
    def __init__(self, agent, graph):
        super().__init__(
            name="find_paths",
            description="Find all the paths from the current node to a given node",
            graph=graph,
        )

    def __call__(self, src_index, dst_index) -> dict[str, Any] | str:
        try:
            src_node = self.graph.get_node_by_index(src_index)
        except Exception as e:
            return f"Node with index {src_index} not found in the graph. Please check the index and try again.\n"
        try:
            dst_node = self.graph.get_node_by_index(dst_index)
        except Exception as e:
            return f"Node with index {dst_index} not found in the graph. Please check the index and try again.\n"
        try:
            paths = self.graph.find_paths_of_length_2(src_node, dst_node)
        except Exception as e:
            paths = []

        # NOTE: THIS SHOULD NEVER HAPPEN. If a node is in the 2-hop neighborhood it means there's at least one path of length 2
        if not paths:
            return f"find_paths({src_node}, {dst_node}) didn't return any results. Consider widening your search or changing the strategy.\n"

        if len(paths) > 30:
            return (
                f"There are too many paths ({len(paths)}) between the current node `{src_node}` and `{dst_node}`. "
                "Showing the first 30 paths. Consider refining your search or narrowing down the nodes of interest.\n"
                f"The first 30 paths between the current node `{src_node}` and `{dst_node}` are\n"
                + "\n".join([str(path) for path in list(paths)[:30]])
                + "\nconsider if any of these paths can help you answer the question\n"
            )

        return (
            f"The paths between the current node `{src_node}` and `{dst_node}` are\n"
            + "\n".join([str(path) for path in paths])
            + "\nconsider if any of these paths can help you answer the question\n"
        )

        return

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dst_node_name": {
                            "type": "string",
                            "description": "The name of the destination node to find paths to",
                        },
                        "dst_node_index": {
                            "type": "integer",
                            "description": "The index of the destination node to find paths to",
                        },
                    },
                    "required": ["dst_node_index"],
                },
            },
        }
