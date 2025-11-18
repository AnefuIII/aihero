import search_tools
from pydantic_ai import Agent
from typing import List, Any
import numpy as np
import openai
import re # <-- NEW: For query cleaning

# --- HELPER FUNCTION ---
def clean_prompt_for_search(prompt: str) -> str:
    """Removes conversational language to improve keyword search precision."""
    # Remove phrases like "tell me", "can you", "what is", "please"
    cleaned_prompt = re.sub(r'\b(give me|tell me|can you tell me|what is|how to|please)\b', '', prompt, flags=re.IGNORECASE)
    # Remove excessive whitespace
    cleaned_prompt = ' '.join(cleaned_prompt.split()).strip()
    return cleaned_prompt if cleaned_prompt else prompt


# --- 1. HYBRID SEARCH TOOL FUNCTION (CORRECTED) ---
class HybridSearchTool:
    def __init__(self, aut_index, autogen_vindex, embedding_model, repo_owner, repo_name):
        self.aut_index = aut_index
        self.autogen_vindex = autogen_vindex
        self.embedding_model = embedding_model
        self.repo_owner = repo_owner
        self.repo_name = repo_name

    def hybrid_search(self, query: str) -> List[Any]:
        """
        Performs a Hybrid Search combining Text (Keyword) and Vector (Semantic) search
        with corrected parameters for minsearch compatibility.
        """
        # 1. Pre-process Query for better keyword matching
        search_query = clean_prompt_for_search(query) 
        
        # 2. Text Search (Keyword matching) - PRIORITIZE KEYWORDS
        # We increase retrieval (k=10) to ensure the specific list is found, 
        # and rely on minsearch's internal scoring for relevance.
        text_results = self.aut_index.search(
            query=search_query, 
            num_results=10 # <-- Increased k further for better precision capture
        )
        
        # 3. Vector Search (Semantic matching) - INCREASED RETRIEVAL (k=8)
        q_vector = self.embedding_model.encode(search_query)
        vector_results = self.autogen_vindex.search(
            q_vector, 
            num_results=8
        )
        
        # 4. Combine, Deduplicate, and Prioritize Results
        seen_contents = set()
        combined_results = []
        
        # Strategy: Process Text results first (high-precision keyword matches), 
        # then fill in with Vector results, up to a limit (MAX_CHUNKS=10).
        MAX_CHUNKS = 10 
        
        # Process Text Results (giving them priority)
        for result in text_results:
            content_id = (result['filename'], result['content'])
            if content_id not in seen_contents and len(combined_results) < MAX_CHUNKS: 
                seen_contents.add(content_id)
                combined_results.append({
                    "filename": result['filename'],
                    "content": result['content']
                })
        
        # Process Vector Results (only if we still have space)
        for result in vector_results:
            content_id = (result['filename'], result['content'])
            if content_id not in seen_contents and len(combined_results) < MAX_CHUNKS: 
                seen_contents.add(content_id)
                combined_results.append({
                # Use the filename for the display text
                "display_name": result['filename'], 
                # Use the URL (which is either the page URL or the PDF URL)
                "link_url": result['url'] if 'url' in result else result['filename'], 
                "content": result['content']
            })
                
        # If no results found
        if not combined_results:
            return [{"filename": "No Results", "content": "The search did not find specific content related to your query in the available documents."}]

        return combined_results


# --- 2. AGENT DEFINITION (MODIFIED) ---

SYSTEM_PROMPT_TEMPLATE = """
You are the FMBN Assistant, an expert RAG system providing accurate information about the Federal Mortgage Bank of Nigeria (FMBN), its policies, and the National Housing Fund (NHF) Act.

**INSTRUCTIONS:**
1.  **Strictly Grounding:** Your response **MUST** be based **ONLY** on the context provided in the `SOURCE DOCUMENTS` below. Do not use external knowledge or make assumptions.
2.  **Citation:** Always provide citations immediately after the information they support. Format the citation as a Markdown link, where the visible text is the source's name and the link destination is the source's URL. Example: [Source Name](Source URL).
3.  **Prioritize Specificity:** When answering a procedural or requirement-based question (e.g., "how to apply," "what documents are needed," "what is the deadline"), you **MUST** prioritize and use:
    * **Specific lists** (e.g., bullet points, numbered lists) found in the context.
    * Content retrieved from sources with highly descriptive titles (e.g., `Official List of Required Documents for NHF Refund`, `FMBN Eligibility Criteria`).
    * Only fall back to general legal text (e.g., the NHF Act) if specific procedural steps are not available.
4.  **Completeness:** Provide a complete answer, addressing all parts of the user's question, while remaining concise.

Always include references by citing the filename of the source material you used.
Example Format: [FMBN Official Homepage]

If the search doesn't return relevant results, let the user know and provide general guidance.
"""

def init_agent(aut_index, autogen_vindex, embedding_model, repo_owner, repo_name, openai_api_key=None):

    if openai_api_key:
        openai.api_key = openai_api_key
    
    # Update the prompt with the new, refined instructions
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(repo_owner=repo_owner, repo_name=repo_name)

    # Initialize the HybridSearchTool
    hybrid_tool = HybridSearchTool(
        aut_index=aut_index, 
        autogen_vindex=autogen_vindex, 
        embedding_model=embedding_model,
        repo_owner=repo_owner, 
        repo_name=repo_name
    )

    # The agent now gets the hybrid_search method as its tool
    agent = Agent(
        name="fmbn_rag_agent", # Renamed for clarity
        instructions=system_prompt,
        tools=[hybrid_tool.hybrid_search], # This is the hybrid search hook
        model='gpt-4o-mini'
    )

    return agent


# text search tool (old, for reference)


# import search_tools
# from pydantic_ai import Agent


# SYSTEM_PROMPT_TEMPLATE = """
# You are a helpful assistant that answers questions about documentation.  

# Use the search tool to find relevant information from the course materials before answering questions.  

# If you can find specific information through search, use it to provide accurate answers.

# Always include references by citing the filename of the source material you used.
# Replace it with the full path to the GitHub repository:
# "https://github.com/{repo_owner}/{repo_name}/blob/main/"
# Format: [LINK TITLE](FULL_GITHUB_LINK)


# If the search doesn't return relevant results, let the user know and provide general guidance.
# """

# def init_agent(index, repo_owner, repo_name):
#     system_prompt = SYSTEM_PROMPT_TEMPLATE.format(repo_owner=repo_owner, repo_name=repo_name)

#     search_tool = search_tools.SearchTool(index=index)

#     agent = Agent(
#         name="gh_agent",
#         instructions=system_prompt,
#         tools=[search_tool.search],
#         model='gpt-4o-mini'
#     )

#     return agent