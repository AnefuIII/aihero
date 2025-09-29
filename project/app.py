# app.py

import streamlit as st
import ingest
import search_agent 
import logs

import asyncio
import os 
from typing import Tuple, Any

# --- GLOBAL CONFIGURATION ---
REPO_OWNER = "microsoft"
REPO_NAME = "autogen"

# ---------------------------------------------------------------------
# Use Streamlit's cache to initialize expensive components ONLY ONCE
# This function handles the entire data ingestion and indexing process.
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="1. Initializing data ingestion (Reading repo, chunking, embedding, indexing)...")
def setup_indexes() -> Tuple[Any, Any, Any]:
    """Runs the indexing process and returns the necessary components."""
    
    # 1. Define the filter function (excluding examples)
    def filter(doc):
        return not doc['filename'].startswith('examples/')
        
    # 2. Call the hybrid indexing function from ingest.py
    aut_index, autogen_vindex, embedding_model = ingest.index_all_data(
        REPO_OWNER, 
        REPO_NAME, 
        filter=filter
    )
    
    return aut_index, autogen_vindex, embedding_model


@st.cache_resource(show_spinner="2. Initializing AI Agent...")
def setup_agent(aut_index, autogen_vindex, embedding_model):
    """Initializes the Agent and attaches the Hybrid Search Tool."""
    
    # NOTE: The Agent automatically reads the OPENAI_API_KEY from the 
    # environment, which Streamlit Cloud populates from st.secrets.
    
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
    
    st.set_page_config(page_title="AutoGen Docs Assistant", layout="wide")
    st.title("🤖 AutoGen Documentation AI Assistant")
    st.markdown("Ask any question about the **AutoGen** repository documentation. The agent uses **Hybrid Search** and **`gpt-4o-mini`** for grounded answers.")

    # 1. Setup the Indexes and Agent
    try:
        aut_index, autogen_vindex, embedding_model = setup_indexes() 
        agent = setup_agent(aut_index, autogen_vindex, embedding_model)
    except Exception as e:
        st.error(f"Failed to initialize the Agent or Indexes. Please check your setup and Streamlit secrets.")
        st.exception(e)
        return # Stop execution if setup fails


    # 2. Initialize Session State for Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hi! I'm ready to answer your questions about the Microsoft AutoGen documentation. What can I help you find?"
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
                
                # Run the agent using asyncio.run
                try:
                    # Note: We use asyncio.run because the Agent's run method is often async.
                    response = asyncio.run(agent.run(user_prompt=prompt))
                    
                    # Get the final text output
                    agent_output = response.output
                    
                    # Log the interaction (using your existing logs module)
                    logs.log_interaction_to_file(agent, response.new_messages())
                    
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