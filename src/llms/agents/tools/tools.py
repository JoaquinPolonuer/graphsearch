from abc import ABC, abstractmethod
from typing import Any

from graph_types.graph import Graph, Node


class Tool(ABC):
    """Abstract base class for all tools"""

    def __init__(self, name: str, description: str, graph: Graph) -> None:
        self.name = name
        self.description = description
        self.graph = graph

    @abstractmethod
    def __call__(self, args: dict[str, Any], current_node: Node, graph: Graph) -> dict[str, Any]:
        """__call__ the tool with given inputs and context"""
        pass

    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """Return the full tool definition for LiteLLM"""
        pass
