# app.py (Corrected)

import streamlit as st
import ingest
import search_agent 
import logs

import asyncio
import os 
from typing import Tuple, Any

# --- GLOBAL CONFIGURATION ---
PDF_FILE_PATH = "cai.pdf" 
REPO_OWNER = "FR" # Used for citation link building
REPO_NAME = "NC" # Used for citation link building
# ---------------------------------------------------------------------

# 1. Setup Indexes (This MUST run only once)
# @st.cache_resource guarantees this function runs only on the first load.
@st.cache_resource(show_spinner="1. Initializing data ingestion (Reading PDF, chunking, embedding, indexing)...")
def setup_indexes() -> Tuple[Any, Any, Any]:
    """Runs the indexing process and returns the necessary components."""
    
    def filter(doc):
        # Always return True for a single PDF to index all content
        return True 
        
    # Call the hybrid indexing function from ingest.py with the PDF path
    aut_index, autogen_vindex, embedding_model = ingest.index_all_data(
        PDF_FILE_PATH,  
        filter=filter
    )
    
    # NOTE: These unhashable objects are cached by Streamlit now.
    return aut_index, autogen_vindex, embedding_model


# 2. Setup Agent (This MUST also run only once)
# This function is dependent on the first one. By calling setup_indexes()
# inside of it, we retrieve the cached results without forcing a rerun 
# of the indexing.

@st.cache_resource(show_spinner="2. Initializing AI Agent...")
def setup_agent():
    """Initializes the Agent and attaches the Hybrid Search Tool.
    
    It retrieves the cached index components from setup_indexes().
    """
    
    # Retrieve the CACHED components from the first function call
    aut_index, autogen_vindex, embedding_model = setup_indexes()
    
    # NOTE: The Agent automatically reads the OPENAI_API_KEY from the 
    # environment/st.secrets.
    
    agent = search_agent.init_agent(
        aut_index, 
        autogen_vindex, 
        embedding_model, 
        REPO_OWNER, 
        REPO_NAME
    )
    return agent

# ---------------------------------------------------------------------
# Streamlit Application Entry Point
# ---------------------------------------------------------------------
def main_app():
    
    st.set_page_config(page_title="Constitution-AI", layout="wide")
    st.title("⚖️ Constitution-AI: Nigerian Constitutional Assistant")
    st.markdown("Ask any question about the **Federal Republic of Nigeria Constitution**. The AI uses **Hybrid Search** and **`gpt-4o-mini`** for grounded answers with article citations.")

    # 1. Setup the Indexes and Agent
    try:
        # We only need to call the setup_agent function, which internally
        # calls and retrieves the CACHED results from setup_indexes.
        agent = setup_agent()
    except Exception as e:
        st.error(f"Failed to initialize the Agent or Indexes. Please check your setup and Streamlit secrets.")
        st.exception(e)
        return # Stop execution if setup fails


    # 2. Initialize Session State for Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hi! I'm ready to answer your questions about the Nigerian Constitution. What can I help you find?"
        })


    # 3. Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # 4. Handle User Input
    if prompt := st.chat_input("Your question:"):
        
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Agent Response Generation
        with st.chat_message("assistant"):
            with st.spinner("Running Hybrid Search & Agent..."):
                
                try:
                    # Run the agent using asyncio.run
                    response = asyncio.run(agent.run(user_prompt=prompt))
                    
                    # Get the final text output
                    agent_output = response.output
                    
                    # Log the interaction (using your existing logs module)
                    # logs.log_interaction_to_file(agent, response.new_messages())
                    
                    # Display the result
                    st.markdown(agent_output)
                    
                    # Add assistant message to state
                    st.session_state.messages.append({"role": "assistant", "content": agent_output})
                    
                except Exception as e:
                    error_message = f"An error occurred during agent execution: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
    main_app()