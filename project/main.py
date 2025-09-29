import ingest
import search_agent 
import logs

import asyncio
import os # Import os to set the API key if needed

# ----------------------------------------------------
from dotenv import load_dotenv
from openai import OpenAI

# This line loads the variables from your .env file
load_dotenv()
openai_client = OpenAI()
# ----------------------------------------------------

REPO_OWNER = "microsoft"
REPO_NAME = "autogen"


def initialize_index():
    print(f"Starting Autogen AI Assistant for {REPO_OWNER}/{REPO_NAME}")
    print("Initializing data ingestion for Hybrid Search...")

    # NOTE ON FILTER: A typical full RAG project searches all docs.
    # If you still need a filter, define it here:
    def filter(doc):
        # Example: Skip files in the 'examples/' folder
        return not doc['filename'].startswith('examples/')
    
    # ASSUMPTION: ingest.py now has a function 'index_all_data' 
    # that returns the three required components for Hybrid Search.
    aut_index, autogen_vindex, embedding_model = ingest.index_all_data(
        REPO_OWNER, 
        REPO_NAME, 
        filter=filter
    )
    
    print("Data indexing completed successfully!")
    
    # Return all three components required for the HybridSearchTool
    return aut_index, autogen_vindex, embedding_model


def initialize_agent(aut_index, autogen_vindex, embedding_model):
    print("Initializing search agent...")
    
    # Pass all three components to the updated init_agent function
    agent = search_agent.init_agent(
        aut_index, 
        autogen_vindex, 
        embedding_model, 
        REPO_OWNER, 
        REPO_NAME
    )
    
    print("Agent initialized successfully!")
    return agent


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

# import ingest
# import search_agent 
# import logs

# import asyncio


# REPO_OWNER = "microsoft"
# REPO_NAME = "autogen"


# def initialize_index():
#     print(f"Starting Autogen AI Assistant for {REPO_OWNER}/{REPO_NAME}")
#     print("Initializing data ingestion...")

#     def filter(doc):
#         return 'data-engineering' in doc['filename']

#     index = ingest.index_data(REPO_OWNER, REPO_NAME, filter=filter)
#     print("Data indexing completed successfully!")
#     return index


# def initialize_agent(index):
#     print("Initializing search agent...")
#     agent = search_agent.init_agent(index, REPO_OWNER, REPO_NAME)
#     print("Agent initialized successfully!")
#     return agent


# def main():
#     index = initialize_index()
#     agent = initialize_agent(index)
#     print("\nReady to answer your questions!")
#     print("Type 'stop' to exit the program.\n")

#     while True:
#         question = input("Your question: ")
#         if question.strip().lower() == 'stop':
#             print("Goodbye!")
#             break

#         print("Processing your question...")
#         response = asyncio.run(agent.run(user_prompt=question))
#         logs.log_interaction_to_file(agent, response.new_messages())

#         print("\nResponse:\n", response.output)
#         print("\n" + "="*50 + "\n")


# if __name__ == "__main__":
#     main()
