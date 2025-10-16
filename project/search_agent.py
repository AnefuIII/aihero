import search_tools
from pydantic_ai import Agent
from typing import List, Any
import numpy as np

# --- 1. HYBRID SEARCH TOOL FUNCTION ---

# NOTE: This function needs access to the indexes and model, 
# so we define a class to hold them, or redefine init_agent 
# to pass them directly. Let's use a class structure 
# similar to your old SearchTool for organization.

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
        on the documentation data.
        """
        # 1. Text Search (Keyword matching)
        text_results = self.aut_index.search(query, num_results=5)
        
        # 2. Vector Search (Semantic matching)
        q_vector = self.embedding_model.encode(query)
        vector_results = self.autogen_vindex.search(q_vector, num_results=5)
        
        # 3. Combine and Deduplicate Results (Simplified to filename for the agent)
        seen_contents = set()
        combined_results = []
        
        for result in text_results + vector_results:
            # We use content + filename for more robust deduplication across different chunks
            content_id = (result['filename'], result['content'])

            # Use content_id for deduplication 
            if content_id not in seen_contents:
                seen_contents.add(content_id)

                # Return the essential information for the LLM
                combined_results.append({
                    "filename": result['filename'],
                    "content": result['content']
                })

        return combined_results


# --- 2. AGENT DEFINITION ---

SYSTEM_PROMPT_TEMPLATE = """
...
Always include references by citing the filename of the source material you used.
Format the reference as a simple citation: [Constitution-of-the-Federal-Republic-of-Nigeria.pdf]
Example Format: [Constitution-of-the-Federal-Republic-of-Nigeria.pdf]

If the search doesn't return relevant results, let the user know and provide general guidance.
"""

def init_agent(aut_index, autogen_vindex, embedding_model, repo_owner, repo_name):
    # Update the prompt to reflect the Hybrid Search tool being used
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(repo_owner=repo_owner, repo_name=repo_name)

    # Initialize the new HybridSearchTool
    hybrid_tool = HybridSearchTool(
        aut_index=aut_index, 
        autogen_vindex=autogen_vindex, 
        embedding_model=embedding_model,
        repo_owner=repo_owner, 
        repo_name=repo_name
    )

    # The agent now gets the hybrid_search method as its tool
    agent = Agent(
        name="gh_agent",
        instructions=system_prompt,
        tools=[hybrid_tool.hybrid_search], # <-- THIS IS THE HYBRID SEARCH HOOK
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