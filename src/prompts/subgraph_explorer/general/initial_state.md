Question: {question}

You are starting your exploration at node: **{node}**

## Your Mission
Find all nodes in this knowledge graph that answer the given question. You have been placed at this starting node because it's strategically relevant to the question.

## Recommended First Steps
1. **Understand your starting point**: Use `search_in_surroundings(k=1, query="")` to see what's immediately connected to your current node
2. **Identify the answer type**: Based on the question, determine what type of nodes you're looking for (papers, genes, products, etc.)
3. **Begin systematic exploration**: Use the available tools to explore the neighborhood and find relevant nodes

## Remember
- Add answers with `add_to_answer` as soon as you find potentially relevant nodes
- You can continue exploring after adding answers
- Better to include more candidates than to miss the correct answer
- Use `find_paths` to understand WHY nodes might be relevant
- You have 10 steps maximum, so explore efficiently

Start exploring!
