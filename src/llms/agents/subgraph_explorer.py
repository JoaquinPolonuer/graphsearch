import json
from typing import Any, Optional

from litellm import completion

from graph_types.graph import Graph, Node
from src.llms.agents.tools.find_paths import FindPathsTool
from src.llms.agents.tools.search_in_surroundings import SearchInSurroundingsTool
from src.llms.agents.tools.submit_answers import SubmitAnswersTool

from src.prompts.prompts import (
    SUBGRAPH_EXPLORER_INITIAL_STATE,
    SUBGRAPH_EXPLORER_SYSTEM,
    MAG_SUBGRAPH_EXPLORER_SYSTEM,
    PRIME_SUBGRAPH_EXPLORER_SYSTEM,
)
from fuzzywuzzy import fuzz
from pydantic import BaseModel, field_validator


def fuzzy_match(name, pattern, threshold=90):
    return fuzz.partial_ratio(name.lower(), pattern.lower()) >= threshold


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

        if "prime" in graph.name:
            system_prompt = PRIME_SUBGRAPH_EXPLORER_SYSTEM.format(node_types=self.graph.node_types)
        elif "mag" in graph.name:
            system_prompt = MAG_SUBGRAPH_EXPLORER_SYSTEM.format(node_types=self.graph.node_types)
        else:
            system_prompt = SUBGRAPH_EXPLORER_SYSTEM

        self.system_prompt = system_prompt.format(node_types=self.graph.node_types)

        self.conversation_as_string = ""

    def get_message(self, response: Any) -> Any:
        return response.choices[0].message

    def select_tools(self) -> Any:
        messages = [{"role": "system", "content": self.system_prompt}] + self.message_history

        try:
            response = completion(
                model=self.model,
                messages=messages,
                tools=[tool.schema() for tool in self.tools],
                tool_choice="required",
            )
        except Exception as e:
            print(f"Error during completion: {e}")
            raise e
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

    def call_tool(self, selected_tool: Any) -> Any:
        function_name = selected_tool.function.name
        args = json.loads(selected_tool.function.arguments)

        for tool in self.tools:
            if tool.name == function_name:
                break

        if function_name == "find_paths":
            return tool(src_index=self.node.index, dst_index=args["dst_node_index"])
        elif function_name == "search_in_surroundings":
            return tool(
                node=self.node, query=args.get("query", ""), type=args.get("type", ""), k=args["k"]
            )
        elif function_name == "submit_answers":
            return tool(agent=self, answer_node_indices=args["answer_node_indices"])
        else:
            raise ValueError(f"Unknown tool: {function_name}")

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
                tool_response = self.call_tool(selected_tool)
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
