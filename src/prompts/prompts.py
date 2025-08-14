# GENERAL SIMPLE CALLS PROMPTS
with open("src/prompts/simple_calls/general/starting_node_filtering_system.md", "r") as f:
    STARTING_NODE_FILTERING_SYSTEM = f.read()

with open("src/prompts/simple_calls/general/entity_extraction_system.md", "r") as f:
    ENTITY_EXTRACTION_SYSTEM = f.read()

with open("src/prompts/simple_calls/general/question_answer_type_system.md", "r") as f:
    QUESTION_ANSWER_TYPE_SYSTEM = f.read()

with open("src/prompts/simple_calls/general/discard_explore_add_system.md", "r") as f:
    DISCARD_EXPLORE_ADD_SYSTEM = f.read()


# MAG SIMPLE CALLS PROMPTS
with open("src/prompts/simple_calls/mag/starting_node_filtering_system.md", "r") as f:
    MAG_STARTING_NODE_FILTERING_SYSTEM = f.read()

# SUBGRAPH EXPLORER PROMPTS
with open("src/prompts/subgraph_explorer/general/initial_state.md", "r") as f:
    SUBGRAPH_EXPLORER_INITIAL_STATE = f.read()

# MAG SUBGRAPH EXPLORER PROMPTS
with open("src/prompts/subgraph_explorer/mag/system_prompt.md", "r") as f:
    MAG_SUBGRAPH_EXPLORER_SYSTEM = f.read()

# PRIME SUBGRAPH EXPLORER PROMPTS
with open("src/prompts/subgraph_explorer/prime/system_prompt.md", "r") as f:
    PRIME_SUBGRAPH_EXPLORER_SYSTEM = f.read()

# AMAZON SUBGRAPH EXPLORER PROMPTS
with open("src/prompts/subgraph_explorer/amazon/system_prompt.md", "r") as f:
    AMAZON_SUBGRAPH_EXPLORER_SYSTEM = f.read()
