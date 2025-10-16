import ingest
import search_agent 
import logs

import asyncio
import os 
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI()

# ----------------------------------------------------
# Define the path to your PDF file
PDF_FILE_PATH = "cai.pdf" # <-- CHANGE THIS to your PDF file
# We'll use a placeholder for REPO_OWNER/REPO_NAME for the link generation, 
# or you can remove them if the agent is no longer expected to generate GitHub links.
REPO_OWNER = "local_project"
REPO_NAME = "pdf_docs"
# # ----------------------------------------------------


def initialize_index():
    # Use the PDF path here instead of the repo info
    print(f"Starting AI Assistant for {PDF_FILE_PATH}")
    print("Initializing data ingestion for Hybrid Search...")

    # NOTE ON FILTER: Filter still works, but now checks the content of the PDF.
    def filter(doc):
        # Example: Skip files if the content contains a specific phrase.
        # For a single PDF, this might not be necessary, so you can return True.
        return True # For now, don't skip any chunks from the PDF
    
    # PASS THE PDF PATH INSTEAD OF REPO OWNER/NAME
    aut_index, autogen_vindex, embedding_model = ingest.index_all_data(
        PDF_FILE_PATH, # <-- MODIFIED: Pass the PDF path as the data source
        filter=filter
    )
    
    print("Data indexing completed successfully!")
    
    return aut_index, autogen_vindex, embedding_model


def initialize_agent(aut_index, autogen_vindex, embedding_model):
    print("Initializing search agent...")
    
    # REPO_OWNER/REPO_NAME are still needed here for the agent's prompt
    # and link generation in search_agent.py, so keep them.
    agent = search_agent.init_agent(
        aut_index, 
        autogen_vindex, 
        embedding_model,
        REPO_OWNER, 
        REPO_NAME
    )
    
    print("Agent initialized successfully!")
    return agent

# The rest of main.py remains the same.


def main():
    # 1. Capture all three returned components from initialize_index
    aut_index, autogen_vindex, embedding_model = initialize_index() 
    
    # 2. Pass all three components to initialize_agent
    agent = initialize_agent(aut_index, autogen_vindex, embedding_model)
    
    print("\nReady to answer your questions!")
    print("Type 'stop' to exit the program.\n")

    while True:
        question = input("Your question: ")
        if question.strip().lower() == 'stop':
            print("Goodbye!")
            break

        print("Processing your question...")
        
        # Note: 'await' only works inside async functions. 
        # Using asyncio.run() is correct for a synchronous main loop.
        response = asyncio.run(agent.run(user_prompt=question))
        
        logs.log_interaction_to_file(agent, response.new_messages())

        print("\nResponse:\n", response.output)
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()
