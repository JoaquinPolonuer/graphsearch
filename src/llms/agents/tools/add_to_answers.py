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
        answer_nodes = [self.graph.get_node_by_index(name) for name in answer_node_indices]
        agent.answer_nodes.extend(answer_nodes)
        return "Added the following nodes to the answer:" + ",".join([str(n) for n in answer_nodes])

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer_node_names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of node names that represent the answer. Pass EVERY node that could be an answer to this question.",
                        },
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
