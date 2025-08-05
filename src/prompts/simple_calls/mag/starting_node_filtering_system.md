## Instructions:

You will be given a question about a graph and a list of nodes from that graph.
The list of nodes was retrieved using a lexical search algorithm, but we need to filter those who are directly related to the question. 
Basically, we need to do the following:
- If there's a node that is specifically mentioned in the question, you should just return that node.
- If no node is mentioned in the question, you should return all nodes that might be connected to the question.

Your task is to select the relevant nodes, so an agent can execute a search in the surroundings of that node.

Return a JSON object with this exact format:
{{
    "relevant_nodes": [
        {{"name": "node_name", "index": node_index}},
        {{"name": "node_name", "index": node_index}}
    ]
}}

## Examples:
Question: 'Can you find me other publications by the co-authors of "Microwave complex permittivity of voltage-tunable nematic liquid crystals measured in high resistivity silicon transducers," specifically those where they discuss developing and testing a new precise method for determining elastic constants in liquid crystals?'
Relevant nodes: [PaperNode(name=Microwave complex permittivity of voltage-tunable nematic liquid crystals measured in high resistivity silicon transducers, index=1403010, type=paper)]

Question:'Show me papers authored by co-authors of the study "Black hole dynamical evolution in a Lorentz-violating spacetime" focusing on early techniques for examining two-electron atoms or similar themes in the field of quantum physics.'
Relevant nodes: [PaperNode(name=Black hole dynamical evolution in a Lorentz-violating spacetime, index=1327472, type=paper)]

Question: 'Are there any Astronomy papers discussing galaxy gas masses linked to Central Connecticut State University on Arxiv?'
Relevant nodes: [InstitutionNode(name=Central Connecticut State University, index=1109346, type=institution)]

Question: 'Is there any research from Tianjin Urban Construction Institute on the slowing of particle beams in dusty plasma within the Dusty plasma field?'
Relevant nodes: [InstitutionNode(name=Tianjin Urban Construction Institute, index=1109798, type=institution)]