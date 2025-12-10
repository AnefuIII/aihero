# app.py — Production-ready for FMBN-AI

import streamlit as st
import ingest
import search_agent
#import asyncio
import os
from typing import Tuple, Any



# ---------------------------
# GLOBAL CONFIGURATION
# ---------------------------
WEBSITE_URL = "https://www.fmbn.gov.ng"
PDF_URL = "https://www.fmbn.gov.ng/documents/NHF_ACT._CAP_N45.pdf"
REPO_OWNER = "FMBN"
REPO_NAME = "Website"
DATA_DIR = "rag_data" # Define the data directory
CHUNK_FILE = os.path.join(DATA_DIR, "chunks.pkl") # Point to the directory
EMB_FILE = os.path.join(DATA_DIR, "embeddings.npy") # Point to the directory
# ...
LOCAL_PDF_DIRECTORY = "pdfs_to_index"

# ---------------------------
# 1️⃣ Setup Indexes (cached)
# ---------------------------
@st.cache_resource(show_spinner="Initializing data (chunks, embeddings, indices)...")
def setup_indexes() -> Tuple[Any, Any, Any]:
    # ...
    aut_index, autogen_vindex, embedding_model = ingest.index_hybrid_data(
        website_url=None,       # <-- Force skip the website
        pdf_url=None,           # <-- Force skip the remote PDF
        local_pdf_dir=LOCAL_PDF_DIRECTORY, 
        chunk_file=CHUNK_FILE,
        emb_file=EMB_FILE,
        max_pages=0,            # To ensure scrape_website_dynamic returns quickly if called
        headless=True
    )
    return aut_index, autogen_vindex, embedding_model

# ---------------------------
# 2️⃣ Setup Agent (cached)
# ---------------------------
@st.cache_resource(show_spinner="Initializing AI Agent...")
def setup_agent():
    aut_index, autogen_vindex, embedding_model = setup_indexes()

    # Read OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        try:
            openai_api_key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            raise RuntimeError("OPENAI_API_KEY not found. Set via env or Streamlit secrets.")

    # Initialize FMBN AI agent
    agent = search_agent.init_agent(
        aut_index=aut_index,
        autogen_vindex=autogen_vindex,
        embedding_model=embedding_model,
        repo_owner=REPO_OWNER,
        repo_name=REPO_NAME,
        openai_api_key=openai_api_key
    )
    return agent

# ---------------------------
# 3️⃣ Streamlit App
# ---------------------------
def main_app():
    st.set_page_config(page_title="FMBN-AI", layout="wide")
    st.title("🏦 FMBN-AI: Federal Mortgage Bank Assistant")
    st.markdown(
        "Ask any question about **FMBN and the NHF Act**. "
        "The AI uses **Hybrid Search** on the website content and the provided **NHF Act PDF document** "
        "for grounded answers with citations."
    )

    # Initialize Agent
    try:
        agent = setup_agent()

        st.session_state.agent = agent
    except Exception as e:
        st.error("Failed to initialize the Agent or Indexes. Check your setup and secrets.")
        st.exception(e)
        return

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hi! I'm ready to answer your questions about FMBN and the NHF Act. What would you like to know?"
        })

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle User Input
    if prompt := st.chat_input("Your question:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Agent Response
        with st.chat_message("assistant"):
            with st.spinner("Running Hybrid Search & Agent..."):
                try:
                    # Run the agent (synchronous)
                    agent = st.session_state.agent
                    agent_output = agent.run(prompt)
                    
                    # Display the result
                    st.markdown(agent_output)
                    
                    # Save assistant message in session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": agent_output
                    })

                except Exception as e:
                    error_message = f"An error occurred during agent execution: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

# ---------------------------
# 4️⃣ Entry Point
# ---------------------------
if __name__ == "__main__":
    main_app()


# # app.py (Corrected)

# import streamlit as st
# import ingest
# import search_agent 
# import logs

# import asyncio
# import os 
# from typing import Tuple, Any

# # # --- GLOBAL CONFIGURATION ---
# # PDF_FILE_PATH = "cai.pdf" 
# # REPO_OWNER = "FR" # Used for citation link building
# # REPO_NAME = "NC" # Used for citation link building
# # # ---------------------------------------------------------------------

# # --- GLOBAL CONFIGURATION (MODIFIED) ---
# WEBSITE_URL = "https://www.fmbn.gov.ng" # <-- NEW
# PDF_URL = "https://www.fmbn.gov.ng/documents/NHF_ACT._CAP_N45.pdf" # <-- NEW
# # NOTE: The citation links will now point to the URLs, not file paths.
# REPO_OWNER = "FMBN" 
# REPO_NAME = "Website" 
# # -----------------------------------

# # 1. Setup Indexes (This MUST run only once)
# # @st.cache_resource guarantees this function runs only on the first load.
# @st.cache_resource(show_spinner="1. Initializing data ingestion (Reading PDF, chunking, embedding, indexing)...")
# def setup_indexes() -> Tuple[Any, Any, Any]:
#     """Runs the indexing process and returns the necessary components."""
    
#     def filter(doc):
#         # Always return True for a single PDF to index all content
#         return True 
        
#     # Call the hybrid indexing function from ingest.py with the PDF path
#     aut_index, autogen_vindex, embedding_model = ingest.index_website_data(
#         WEBSITE_URL,  
#         PDF_URL,
#         filter=filter
#     )
    
#     # NOTE: These unhashable objects are cached by Streamlit now.
#     return aut_index, autogen_vindex, embedding_model

#     # NOTE: These unhashable objects are cached by Streamlit now.
#     return aut_index, autogen_vindex, embedding_model


# # 2. Setup Agent (This MUST also run only once)
# # This function is dependent on the first one. By calling setup_indexes()
# # inside of it, we retrieve the cached results without forcing a rerun 
# # of the indexing.
# # 2. Setup Agent (This MUST also run only once)
# # ...

# # project/app.py (Modified setup_agent function)

# # ... (inside setup_agent function)

# @st.cache_resource(show_spinner="2. Initializing AI Agent...")
# def setup_agent():
    
#     aut_index, autogen_vindex, embedding_model = setup_indexes()
    
#     openai_api_key = None
    
#     # 1. Prioritize reading from the environment variable (for local 'export')
#     if "OPENAI_API_KEY" in os.environ:
#         openai_api_key = os.environ["OPENAI_API_KEY"]
    
#     # 2. If key is NOT found in os.environ, safely attempt to read from st.secrets
#     if not openai_api_key:
#         try:
#             # This line will only run if openai_api_key is still None
#             # We use a try/except to gracefully handle the StreamlitSecretNotFoundError 
#             # that occurs when the app is run locally without a secrets.toml file.
#             openai_api_key = st.secrets["OPENAI_API_KEY"]
#         except Exception:
#             # If any exception occurs (like the StreamlitSecretNotFoundError), 
#             # we ignore it and leave openai_api_key as None.
#             pass

#     # 3. If still not found, raise a clear error
#     if not openai_api_key:
#          raise RuntimeError(
#              "OPENAI_API_KEY not found. Set it via 'export' locally "
#              "or in Streamlit secrets for deployment."
#          )
    
#     # Pass the key to the agent creation logic
#     agent = search_agent.init_agent(
#         aut_index, 
#         autogen_vindex, 
#         embedding_model, 
#         REPO_OWNER, 
#         REPO_NAME,
#         openai_api_key=openai_api_key 
#     )
#     return agent

# # ---------------------------------------------------------------------
# # Streamlit Application Entry Point
# # ---------------------------------------------------------------------
# def main_app():
    
#     # st.set_page_config(page_title="Constitution-AI", layout="wide")
#     # st.title("⚖️ Constitution-AI: Nigerian Constitutional Assistant")
#     # st.markdown("Ask any question about the **Federal Republic of Nigeria Constitution**. The AI uses **Hybrid Search** and **`gpt-4o-mini`** for grounded answers with article citations.")
    
#     st.set_page_config(page_title="FMBN-AI", layout="wide")
#     st.title("🏦 FMBN-AI: Federal Mortgage Bank Assistant")
#     st.markdown(
#         "Ask any question about **FMBN and the NHF Act**. The AI uses **Hybrid Search** "
#         "on the website content and the provided **NHF Act PDF document** for grounded answers with citations." # <-- MODIFIED
#     )
#     # 1. Setup the Indexes and Agent
#     try:
#         # We only need to call the setup_agent function, which internally
#         # calls and retrieves the CACHED results from setup_indexes.
#         agent = setup_agent()
#     except Exception as e:
#         st.error(f"Failed to initialize the Agent or Indexes. Please check your setup and Streamlit secrets.")
#         st.exception(e)
#         return # Stop execution if setup fails


#     # 2. Initialize Session State for Chat History
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#         st.session_state.messages.append({
#             "role": "assistant", 
#             "content": "Hi! I'm ready to answer your questions about the Nigerian Constitution. What can I help you find?"
#         })


#     # 3. Display Chat History
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])


#     # 4. Handle User Input
#     if prompt := st.chat_input("Your question:"):
        
#         # Add user message to state and display
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Agent Response Generation
#         with st.chat_message("assistant"):
#             with st.spinner("Running Hybrid Search & Agent..."):
                
#                 try:
#                     # Run the agent using asyncio.run
#                     response = asyncio.run(agent.run(user_prompt=prompt))
                    
#                     # Get the final text output
#                     agent_output = response.output
                    
#                     # Log the interaction (using your existing logs module)
#                     # logs.log_interaction_to_file(agent, response.new_messages())
                    
#                     # Display the result
#                     st.markdown(agent_output)
                    
#                     # Add assistant message to state
#                     st.session_state.messages.append({"role": "assistant", "content": agent_output})
                    
#                 except Exception as e:
#                     error_message = f"An error occurred during agent execution: {e}"
#                     st.error(error_message)
#                     st.session_state.messages.append({"role": "assistant", "content": error_message})


# if __name__ == "__main__":
#     main_app()