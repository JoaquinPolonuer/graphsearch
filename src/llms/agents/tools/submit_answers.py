from typing import Any
from src.llms.agents.tools.tools import Tool
from graph_types.graph import Graph, Node

class SubmitAnswersTool(Tool):
    def __init__(self, agent, graph):
        super().__init__(
            name="submit_answers",
            description="Submit the final answer based on the current exploration",
            graph=graph,
        )

    def __call__(self, agent, answer_node_indices) -> str:
        answer_nodes = [self.graph.get_node_by_index(name) for name in answer_node_indices]
        agent.final_answer = answer_nodes
        return ",".join([str(n) for n in answer_nodes])

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

    def submit_answer(self, answer: list[Node]) -> list[Node]:
        self.final_answer = answer
        return answer
