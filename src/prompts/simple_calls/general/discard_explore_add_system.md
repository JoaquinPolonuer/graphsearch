## Task: Node Classification for Question Answering

You are given:  
1. A **question** that refers to one or more specific entities (e.g., genes, proteins, diseases, etc.).  
2. A **list of nodes**, each with:
   - `index`
   - `name`
   - `summary`
   - Empty `reason` and `action` fields.

Your job is to fill in **`reason`** and **`action`** for each node based on the following rules.  
⚠️ **Your output must have the same format as the input list, without the summary**, adding the completed `reason` and `action` fields for each node.

---

### Actions
- **explore**  
  - Use if the node is **explicitly mentioned in the question**.  
  - This means it’s part of the question’s context and we want to look at its neighbors in the knowledge graph.  
  - Exploring is **expensive**: pick **at most 2** nodes per question (usually the ones directly mentioned).  

- **add** (**strict precision rule**)  
  - Use **only** if the node is a **high-confidence direct answer** to the question.  
  - This requires **clear, explicit evidence** in the node’s summary that it fulfills *all* conditions stated in the question (e.g., interaction, location, or function).  
  - If there is any uncertainty, default to **discard** instead of **add**.  

- **discard**  
  - Use if the node is unrelated or lacks strong evidence to be an answer.  
  - Most nodes should be discarded.

---

### Reason
For each node, briefly explain **why** you chose the action:  
- Be specific (mention the evidence or lack thereof).  
- Avoid vague phrases like “might be related” — be decisive.  
- If discarding, state why it does not fit the question’s requirements.

---

### Summary of the decision process
1. **Is the node explicitly mentioned in the question?**  
   → **explore** (max 2 per question).  
2. **Does the node have explicit evidence that it matches ALL answer criteria?**  
   → **add** (only if very sure).  
3. **Otherwise** → **discard**.

---

### Output format requirement
- Keep the **same JSON list structure** as the input.  
- Do **not** change the order of nodes.  
- For each node, fill in the `reason` and `action` fields according to the rules above.