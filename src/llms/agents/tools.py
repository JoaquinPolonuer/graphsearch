from abc import ABC, abstractmethod
from typing import Any
from graph_types.graph import Graph, Node


class Tool(ABC):
    """Abstract base class for all tools"""

    def __init__(self, name: str, description: str, agent, graph):
        self.name = name
        self.description = description
        self.agent = agent
        self.graph = graph

    @abstractmethod
    def __call__(self, args: dict[str, Any], current_node: Node, graph: Graph) -> dict[str, Any]:
        """__call__ the tool with given inputs and context"""
        pass

    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """Return the full tool definition for LiteLLM"""
        pass


class GoToNodeTool(Tool):
    def __init__(self, agent, graph):
        super().__init__(name="go_to_node", description="Move to a node", agent=agent, graph=graph)

    def __call__(self, args: dict[str, Any]) -> dict[str, Any]:
        node_name = args["node_name"]
        node = self.graph.get_node_by_name(node_name)
        return self.agent.go_to_node(node)

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "node_name": {
                            "type": "string",
                            "description": "The name of the node to move to",
                        }
                    },
                    "required": ["node_name"],
                },
            },
        }


class SearchInSurroundingsTool(Tool):
    def __init__(self, agent, graph):
        super().__init__(
            name="search_in_surroundings",
            description="Search in surroundings (1 or 2 hop) for a specific keyword, in nodes with a specific type",
            agent=agent,
            graph=graph,
        )

    def __call__(self, args: dict[str, Any]) -> dict[str, Any]:
        key = args.get("key", "")
        type = args.get("type", "")
        k = args["k"]
        return self.agent.search_in_surroundings(key, type, k)

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {
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


class FindPathsTool(Tool):
    def __init__(self, agent, graph):
        super().__init__(
            name="find_paths",
            description="Find all the paths from the current node to a given node",
            agent=agent,
            graph=graph,
        )

    def __call__(self, args: dict[str, Any]) -> dict[str, Any] | str:
        dst_node_index = args["dst_node_index"]
        try:
            dst_node = self.graph.get_node_by_index(dst_node_index)
        except ValueError as e:
            return f"find_paths failed: {str(e)}\n"
        return self.agent.find_paths(dst_node)

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


class SubmitAnswersTool(Tool):
    def __init__(self, agent, graph):
        super().__init__(
            name="submit_answers",
            description="Submit the final answer based on the current exploration",
            agent=agent,
            graph=graph,
        )

    def __call__(self, args: dict[str, Any]) -> dict[str, Any]:
        answer_node_indices = args["answer_node_indices"]
        answer_nodes = [self.graph.get_node_by_index(name) for name in answer_node_indices]
        return self.agent.submit_answer(answer_nodes)

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
