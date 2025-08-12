# Claude's Repository Analysis Summary

## Project Overview
This is a graph-based Retrieval-Augmented Generation (RAG) system for complex question answering over knowledge graphs. The system uses LLM agents with function calling to explore knowledge graphs and collect relevant context for answering questions.

## Core Architecture

### Agent-Based Exploration
- **Main Agent**: `SubgraphExplorerAgent` - Explores knowledge graphs using tool-based function calling
- **Starting Point**: Agent begins at a strategically chosen node and explores the neighborhood
- **Tool-Based**: Uses three main tools to explore and collect information
- **Step Limit**: Maximum 10 exploration steps per question

### Available Tools
1. **search_in_surroundings**: Search 1-hop or 2-hop neighborhoods
   - Supports keyword queries or blank queries (shows all neighbors)
   - Type filtering available
   - Results limited to 15 nodes per call
2. **find_paths**: Find connection paths between current node and destination
   - Uses node indices only (cleaned up for precision)
   - Shows paths up to length 2
3. **add_to_answer**: Collect relevant nodes as potential answers
   - Agents should add answers early and often
   - Can continue exploring after adding answers

### Supported Datasets
- **Amazon**: Product recommendations and shopping queries
- **MAG (Microsoft Academic Graph)**: Academic papers and research relationships  
- **PRIME**: Biomedical knowledge graph (drugs, genes, diseases, phenotypes)

## Key Implementation Details

### Dataset-Specific System Prompts
Each dataset has a tailored system prompt with specific strategies:
- **Location**: `src/prompts/subgraph_explorer/{dataset}/system_prompt.md`
- **MAG**: Focus on papers-only answers, author collaboration networks
- **PRIME**: Biomedical entity relationships, drug-target-disease pathways
- **Amazon**: Product similarity, feature matching, brand preferences

### Recall-Focused Design
Recent improvements emphasize recall over precision:
- **Blank query sweeps**: Start with empty queries to see all neighbors
- **Multiple search angles**: Try synonyms, variants, different phrasings
- **Early candidate capture**: Add potential answers immediately
- **Breadth-first exploration**: Wide coverage before deep dives
- **Lenient inclusion**: "When in doubt, include"

### Code Structure
- **Agents**: `src/llms/agents/subgraph_explorer.py`
- **Tools**: `src/llms/agents/tools/`
- **Prompts**: `src/prompts/prompts.py` (loads all prompt files)
- **Graph Types**: Custom graph implementation with node/edge management

## Prompt Engineering Insights

### Common Question Patterns
Each dataset has identified question pattern categories:
- **MAG**: Co-authorship, institution-based, citation networks, topical research
- **PRIME**: Drug discovery, gene interactions, disease associations, pathway analysis
- **Amazon**: Product similarity, feature-specific, brand searches, budget considerations

### Critical Strategies
- **Progressive escalation**: k=1 → k=2, specific → general queries
- **Type diversity**: Don't limit to one node type unless certain
- **Handle truncation**: Watch for 15-node result limits
- **Multi-angle search**: Synonyms, abbreviations, partial matches

## Recent Changes Made

### System Prompt Improvements
1. **Removed general prompt**: Now uses only dataset-specific prompts
2. **Added Amazon prompt**: Complete with product recommendation strategies
3. **Enhanced recall focus**: Incorporated GPT-5 insights for better recall
4. **Updated all datasets**: MAG, PRIME, and Amazon all have recall-focused strategies

### Tool Cleanup
- **Simplified parameters**: Removed ambiguous name/index dual parameters
- **Index-only approach**: Uses precise node indices to avoid matching errors
- **Cleaner schemas**: Simplified tool function signatures

### Code Updates
- **Prompt loading**: Updated `prompts.py` to load Amazon prompt
- **Agent logic**: Added Amazon dataset support with proper error handling
- **Import cleanup**: Removed unused general prompt imports

## Development Notes

### Testing Commands
- Look for test scripts in README or search codebase for test commands
- No specific lint/typecheck commands identified - ask user if needed

### Performance Considerations
- **Step limit**: 10 exploration steps maximum per question
- **Result truncation**: Search results limited to 15 nodes per call
- **Path complexity**: Shows up to 30 paths, prioritizes shorter ones

### Future Improvements
- Consider adding "reasoning" parameter to tools for agent explainability
- Dataset expansion possible with new system prompts
- Potential for dynamic starting node selection improvements

## Key Files Modified in This Session
- `src/prompts/subgraph_explorer/general/system_prompt.md` (removed)
- `src/prompts/subgraph_explorer/mag/system_prompt.md` (enhanced)
- `src/prompts/subgraph_explorer/prime/system_prompt.md` (enhanced)
- `src/prompts/subgraph_explorer/amazon/system_prompt.md` (created)
- `src/prompts/prompts.py` (updated imports)
- `src/llms/agents/subgraph_explorer.py` (added Amazon support)

This system represents a sophisticated approach to knowledge graph exploration with strong focus on recall optimization and dataset-specific strategies.