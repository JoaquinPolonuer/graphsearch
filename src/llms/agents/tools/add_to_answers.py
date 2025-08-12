from typing import Any
from src.llms.agents.tools.tools import Tool
from graph_types.graph import Graph, Node


class AddToAnswer(Tool):
    def __init__(self, graph):
        super().__init__(
            name="add_to_answer",
            description="Add nodes to the answer.",
            graph=graph,
        )

    def __call__(self, agent, answer_node_indices) -> str:
        answer_nodes = []
        error_indices = []
        for index in answer_node_indices:
            try:
                answer_nodes.append(self.graph.get_node_by_index(index))
            except ValueError:
                error_indices.append(index)

        agent.answer_nodes.extend(answer_nodes)
        res = "Added the following nodes to the answer:" + ",".join([str(n) for n in answer_nodes])
        if error_indices:
            res += f"\nThe following indices were not found in the graph: {error_indices}"

        return res

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer_node_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of node indices that represent the answer. Pass EVERY node that could be an answer to this question.",
                        },
                    },
                    "required": ["answer_node_indices"],
                },
            },
        }
