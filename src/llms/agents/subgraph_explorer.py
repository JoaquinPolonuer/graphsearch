import json
from typing import Any, Optional

from litellm import completion

from graph_types.graph import Graph, Node
from src.llms.agents.tools import (
    FindPathsTool,
    SearchInSurroundingsTool,
    SubmitAnswersTool,
)
from src.prompts.prompts import (
    SUBGRAPH_EXPLORER_INITIAL_STATE,
    SUBGRAPH_EXPLORER_SYSTEM,
)


class SubgraphExplorerAgent:

    def __init__(
        self,
        node: Node,
        graph: Graph,
        question: str,
        model: str = "azure/gpt-4o-1120",
    ) -> None:
        self.model = model
        self.message_history = []
        self.tools = [
            SearchInSurroundingsTool(self, graph),
            FindPathsTool(self, graph),
            SubmitAnswersTool(self, graph),
        ]

        self.graph = graph
        self.node = node
        self.question = question

        self.path = []

        self.final_answer = None

        self.system_prompt = SUBGRAPH_EXPLORER_SYSTEM.format(
            node_types=self.graph.node_types
        )

        self.conversation_as_string = ""

    def get_message(self, response: Any) -> Any:
        return response.choices[0].message

    def search_in_surroundings(self, query: str, type: Optional[str], k: int) -> str:
        if not k in [1, 2]:
            return f"search_in_surroundings({query}, {type}, {k}) only supports k=1 or k=2.\n"

        candidates = self.graph.simple_search_in_surroundings(
            self.node, query=query, type=type, k=k
        )

        if not candidates:
            return f"search_in_surroundings({query}, {type}, {k}) didn't return any results. Consider widening your search or changing the strategy.\n"

        if len(candidates) > 20:
            return (
                f"search_in_surroundings({query}, {type}, {k}) returned too many candidates ({len(candidates)}), showing first 20. "
                "Consider searching for more specific terms or filter by type.\n"
                + "\n".join([str(node) for node in candidates[:20]])
                + "\nIf you found what you were looking for you may want to find the connections between the current node "
                f"`{self.node.name}` and any of the candidate nodes by using the `find_paths` function.\n"
            )

        return (
            f"search_in_surroundings({query}, {type}, {k}) gave the following results\n"
            + "\n".join([str(node) for node in candidates])
            + "\nIf you found what you were looking for you may want to find the connections between the current node "
            f"`{self.node.name}` and any of the candidate nodes by using the `find_paths` function.\n"
        )

    def find_paths(self, node: Node) -> str:
        paths = self.graph.find_paths_of_length_2(self.node, node)

        if not paths:
            return f"find_paths({self.node.name}, {node.name}) didn't return any results. Consider widening your search or changing the strategy.\n"

        if len(paths) > 30:
            return (
                f"There are too many paths ({len(paths)}) between the current node `{self.node.name}` and `{node.name}`. "
                "Showing the first 30 paths. Consider refining your search or narrowing down the nodes of interest.\n"
                f"The first 30 paths between the current node `{self.node.name}` and `{node.name}` are\n"
                + "\n".join([str(path) for path in list(paths)[:30]])
                + "\nconsider if any of these paths can help you answer the question\n"
            )

        return (
            f"The paths between the current node `{self.node.name}` and `{node.name}` are\n"
            + "\n".join([str(path) for path in paths])
            + "\nconsider if any of these paths can help you answer the question\n"
        )

    def submit_answer(self, answer: list[Node]) -> list[Node]:
        self.final_answer = answer
        return answer

    def select_tools(self) -> Any:
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.message_history

        response = completion(
            model=self.model,
            messages=messages,
            tools=[tool.schema() for tool in self.tools],
            tool_choice="required",
        )

        assistant_message = {"role": "assistant"}

        if self.get_message(response).content:
            assistant_message = {
                "role": "assistant",
                "content": self.get_message(response).content,
            }
        if self.get_message(response).tool_calls:
            assistant_message = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in self.get_message(response).tool_calls
                ],
            }

        self.message_history.append(assistant_message)

        return self.get_message(response).tool_calls

    def submit_tool_response(self, tool_response: str, tool_call_id: str) -> None:
        self.message_history.append(
            {"role": "tool", "content": tool_response, "tool_call_id": tool_call_id}
        )

    def get_tool(self, selected_tool: Any) -> Any:
        for tool in self.tools:
            if tool.name == selected_tool.function.name:
                return tool

    def answer(self) -> list[Node]:
        state = SUBGRAPH_EXPLORER_INITIAL_STATE.format(
            question=self.question,
            node=self.node.name,
        )

        print(state)
        self.conversation_as_string += state + "\n"
        self.message_history.append({"role": "user", "content": state})

        for i in range(10):
            selected_tools = self.select_tools()

            for selected_tool in selected_tools:
                tool = self.get_tool(selected_tool)

                arguments = json.loads(selected_tool.function.arguments)
                tool_response = tool(arguments)

                self.submit_tool_response(tool_response, selected_tool.id)
                print(
                    tool_response,
                    "\n",
                    "================================================",
                )
                self.conversation_as_string += str(tool_response) + "\n"

            if self.final_answer:
                return self.final_answer

        return []
