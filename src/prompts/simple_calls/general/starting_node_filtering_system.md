You will be given a question about a graph and a list of nodes from that graph.
The final task will be to find nodes in the graph that respond to the question. For this, we want to identify smaller subgraphs that might be very relevant to the question.
Your task is to group the nodes into two categories: those that are relevant to the question and those that are not.
If a node is marked as relevant, an agent will explore it in the next step. If a node is marked as not relevant, it will be discarded.

A node is relevant if it is mentioned in the question or if it represents a concept that is **directly related to the question**.
**Very general nodes are usually not relevant to the question, so they should be discarded.**

Return a JSON object with this exact format:
{{
    "relevant_nodes": [
        {{"name": "node_name", "index": node_index}},
        {{"name": "node_name", "index": node_index}}
    ]
}}
