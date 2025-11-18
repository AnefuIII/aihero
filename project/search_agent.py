# search_agent.py

import re
import openai
from typing import List, Dict, Any
from ingest import Index, VectorSearch
from sentence_transformers import SentenceTransformer
from pydantic_ai import Agent  # Wraps hybrid search as a tool

# ---------------------------
# Helper: Clean prompt for keyword search
# ---------------------------
def clean_prompt_for_search(prompt: str) -> str:
    """Remove conversational phrases for better keyword search precision."""
    cleaned_prompt = re.sub(
        r'\b(give me|tell me|can you tell me|what is|how to|please)\b',
        '',
        prompt,
        flags=re.IGNORECASE
    )
    cleaned_prompt = ' '.join(cleaned_prompt.split()).strip()
    return cleaned_prompt if cleaned_prompt else prompt

# ---------------------------
# Hybrid Search Tool
# ---------------------------
class HybridSearchTool:
    def __init__(
        self, 
        aut_index: Index, 
        autogen_vindex: VectorSearch, 
        embedding_model: SentenceTransformer, 
        repo_owner: str, 
        repo_name: str
    ):
        self.aut_index = aut_index
        self.autogen_vindex = autogen_vindex
        self.embedding_model = embedding_model
        self.repo_owner = repo_owner
        self.repo_name = repo_name

    def hybrid_search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Combines text (keyword) search and vector (semantic) search.
        Deduplicates results and ensures URLs are preserved.
        """
        search_query = clean_prompt_for_search(query)

        # Text search
        text_results = self.aut_index.search(query=search_query, num_results=top_k)

        # Vector search
        q_vector = self.embedding_model.encode(search_query)
        vector_results = self.autogen_vindex.search(q_vector, num_results=top_k)

        # Combine and deduplicate
        seen = set()
        combined_results = []

        for r in text_results + vector_results:
            key = (r.get("filename", ""), r.get("content", ""))
            if key not in seen and len(combined_results) < top_k:
                seen.add(key)
                combined_results.append({
                    "source_name": r.get("filename", "Unknown Source"),
                    "content": r.get("content", ""),
                    "url": r.get("url", r.get("filename", "N/A"))
                })

        if not combined_results:
            combined_results.append({
                "source_name": "No Results",
                "content": "No relevant content found for your query in the available documents.",
                "url": ""
            })

        return combined_results


# ---------------------------
# FMBN Agent Class
# ---------------------------
from openai import OpenAI

class FMBNAgent:
    def __init__(self, hybrid_tool: HybridSearchTool, openai_api_key: str):
        self.hybrid_tool = hybrid_tool
        self.client = OpenAI(api_key=openai_api_key)

    def run(self, user_prompt: str) -> str:
        # Retrieve top chunks
        chunks = self.hybrid_tool.hybrid_search(user_prompt, top_k=15)

        sources_text = ""
        for i, c in enumerate(chunks, start=1):
            url = c.get("url", "")
            source_name = c.get("source_name", "Unknown Source")
            sources_text += f"{i}. {c['content']}\n([Source: {source_name}]({url}))\n\n"

        prompt = (
            f"Answer the user's question ONLY using the sources below. "
            f"Include citations immediately after each fact, formatted as Markdown links.\n\n"
            f"SOURCES:\n{sources_text}\n"
            f"QUESTION: {user_prompt}\n"
            f"ANSWER:"
        )

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )

        return response.choices[0].message.content

# ---------------------------
# System Prompt Template
# ---------------------------
SYSTEM_PROMPT_TEMPLATE = """
You are the FMBN Assistant, an expert RAG system providing accurate information about
the Federal Mortgage Bank of Nigeria (FMBN) and the National Housing Fund (NHF) Act.

INSTRUCTIONS:
1. STRICTLY ground your answers ONLY in the provided SOURCE DOCUMENTS.
2. Always provide citations immediately after the information you use. Format as a Markdown link: [Source Name](URL).
3. Prioritize specific lists, procedures, or descriptive content from the documents.
4. Provide complete, concise answers addressing all parts of the user's question.
5. If a question is not answered in the documents, respond politely that the information is not available.
"""

# ---------------------------
# Unified Agent Initialization
# ---------------------------
def init_agent(
    aut_index: Index,
    autogen_vindex: VectorSearch,
    embedding_model: SentenceTransformer,
    repo_owner: str,
    repo_name: str,
    openai_api_key: str
) -> FMBNAgent:
    """
    Initializes the FMBN RAG agent with hybrid search and OpenAI API.
    """
    hybrid_tool = HybridSearchTool(
        aut_index=aut_index,
        autogen_vindex=autogen_vindex,
        embedding_model=embedding_model,
        repo_owner=repo_owner,
        repo_name=repo_name
    )

    agent = FMBNAgent(
        hybrid_tool=hybrid_tool,
        openai_api_key=openai_api_key
    )

    return agent
