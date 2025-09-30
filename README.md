# AI Hero

This project, developed as part of a 7-day course, builds a conversational agent that can answer questions about GitHub repository documentation. This repository showcases my learning journey and code.

---

### Day 1: Ingest and Index Your Data

**Objective:** On Day 1, my primary goal was to set up the project environment and build a data ingestion pipeline to process documentation from a chosen GitHub repository.

**What I Learned & Accomplished:**

* **Project Setup:** I successfully initialized my personal Git repository (`aihero`) and structured it with dedicated `course` and `project` folders as per the curriculum.
* **Data Ingestion:** I developed a Python function to download documentation from the **`microsoft/autogen`** repository. I learned the course's recommended method of processing data from a zip archive directly in memory, avoiding the need for a full `git clone`.
* **Data Parsing:** I gained practical experience using the `python-frontmatter` library to parse Markdown documents, extracting structured metadata and the main content.
* **Git Mastery:** I debugged a key issue with nested Git repositories, solidifying my understanding of core Git commands like `git init`, `git add`, `git commit`, and `git branch -M`. I now have a clean, working repository for my project.

**Next Steps:** Tomorrow, I will be learning about chunking large documents to improve search relevance and prepare the data for our AI agent.

### Day 2: Data Ingestion and Chunking for AI Agents

**Objective:** On Day 2, I expanded my data ingestion pipeline to handle a broader range of file types from the Data Engineering Zoomcamp repository, and crucially, began preparing this data for efficient use by an AI agent through a technique called chunking.

**What I Learned & Accomplished:**

* **Extended Data Pipeline:**
    * I enhanced the `read_repo_data` function to download and parse a wider array of file types from GitHub, including Markdown (`.md`, `.mdx`), Python (`.py`), SQL (`.sql`), Java (`.java`), and Jupyter Notebooks (`.ipynb`). This is vital for comprehensive knowledge base creation from diverse repositories.
    * For documentation files, `frontmatter` is used to extract metadata and content. For code files, the raw content is preserved with a `code: True` flag.
    * This refined pipeline successfully ingested 192 documents from the `DataTalksClub/data-engineering-zoomcamp` repository.

* **Jupyter Notebook Processing:**
    * I implemented `format_notebook_as_md` using `nbformat` and `nbconvert` to transform Jupyter Notebook content into a cleaner Markdown format, stripping outputs for a focused documentation view.
    * The `strip_code_fence` helper function further cleans up extracted content.

* **LLM-Powered Content Enrichment:**
    * I leveraged the OpenAI API (`gpt-4o-mini`) with specific instructions to process both notebooks and general code files.
    * `notebook_editing_instructions`: This prompt guides the LLM to convert cleaned notebook Markdown into well-structured documentation, adding section headers and concise, high-level explanations for code blocks.
    * `code_doc_instructions`: This prompt directs the LLM to analyze raw code, describing its purpose, functions, and overall script flow, producing clear, high-level documentation.
    * **Crucial Insight:** While the initial run showed "processing 0 jupyter notebooks...", the pipeline is set up to apply these LLM transformations to actual Jupyter Notebooks and 78 other code files once they are correctly identified and passed. This demonstrates the power of using LLMs for automated documentation generation.

* **Introduction to Chunking:**
    * I implemented a `sliding_window` function to break down large documents into smaller, manageable `chunks` of a specified `size` (2000 characters) with a `step` (1000 characters) of overlap.
    * This process resulted in 865 chunks from the initial 192 documents, demonstrating the practical application of chunking for "large" documents. Each chunk now includes its `start` position within the original document and all the original document's metadata.
    * **Learning:** Chunking is essential for search relevance, improving AI model performance, and respecting token limits, especially for extensive documentation like the Data Engineering Zoomcamp materials.

* **Output and Persistence:** The processed and chunked data is saved into a JSON file (`data/de-zoomcamp-processed.json`), ready for further ingestion into a search engine.

**Next Steps:** Tomorrow, I anticipate focusing on integrating this processed data into a search engine, making it accessible to our AI agent.

---

### Day 3: Hybrid Search and Index Creation

**Objective:** On Day 3, I integrated the chunked data into a production-ready search system, implementing a **Hybrid Search** strategy that is more effective than simple vector search alone.

**What I Learned & Accomplished:**

* **Index Integration:** I utilized a lightweight search library (`minsearch` for this project) to create two distinct indexes from the chunked data:
    * **Keyword Index (`aut_index`):** A standard inverted index based on TF-IDF (Term Frequency-Inverse Document Frequency) is used to capture exact keyword matches. This ensures that a search for a specific function name or filename is highly accurate.
    * **Vector Index (`autogen_vindex`):** This index stores the vector embeddings of each chunk's content, enabling **semantic search**. This allows the RAG system to find relevant answers even if the user uses synonyms or asks conceptual questions.
* **Embedding Model Integration:** I configured the use of an OpenAI embedding model to transform the text chunks into high-dimensional vectors, which is fundamental for the vector index creation.
* **The Hybrid Advantage:** By combining keyword (sparse) and vector (dense) search, the system mitigates the weaknesses of each individual method, providing a much more robust and relevant set of documents for the AI agent.
* **Persistence:** The final indexing logic (`ingest.py`) was confirmed to save the necessary index components, ready to be loaded instantly by the agent.

**Next Steps:** With the data indexed, the next step is to use these indexes to build the actual AI agent that can understand the search results and formulate a final answer.

### Day 4: Building the Conversational Agent and RAG Tool

**Objective:** Day 4 focused on creating the core of the RAG application: an AI Agent capable of interacting with the user and utilizing the newly created Hybrid Search index as a **tool**.

**What I Learned & Accomplished:**

* **Agent Architecture:** I set up the initial agent framework, configuring it with a robust system prompt to define its role as an "expert assistant for AutoGen documentation."
* **Tool Creation (`HybridSearchTool`):** This was the most critical step. I developed a dedicated function that accepts a user's query, executes the **Hybrid Search** across both the keyword and vector indexes, and retrieves the top relevant document chunks.
* **RAG Flow Implementation:** The agent is given explicit instructions to use the `HybridSearchTool` first for every user query. The sequence of operations is now:
    1. User asks a question.
    2. The Agent calls the `HybridSearchTool` with the question.
    3. The tool returns a list of source document chunks.
    4. The Agent takes the original question **and** the retrieved source material, then generates a final, grounded answer.
* **Agent Initialization:** I finalized the `initialize_agent` function, which binds the Hybrid Search indexes to the agent at runtime, ensuring the agent always has access to the most up-to-date knowledge base.

**Next Steps:** The agent can now answer questions, but the interface is a simple command line. Day 5 is dedicated to building a user-friendly web interface using Streamlit.

### Day 5: Streamlit Web Interface Development

**Objective:** To move the conversational agent from the command line to a modern, interactive web application using Streamlit.

**What I Learned & Accomplished:**

* **Streamlit Setup:** I created the `app.py` file and configured the necessary Streamlit components, including session state (`st.session_state`) to maintain the conversation history between user turns.
* **UI Design:** I implemented a clean chat interface using `st.chat_message` and `st.chat_input`. The design ensures that all previous messages are displayed, providing a natural conversational flow.
* **Agent Integration:** I successfully adapted the core agent logic (which was previously in `main.py`) to the Streamlit environment. Crucially, I implemented the `@st.cache_resource` decorator to wrap the index and agent initialization, ensuring that the heavy lifting (loading indexes, initializing the LLM client) only occurs once when the app starts, rather than on every user interaction.
* **Source Attribution:** I enhanced the final response display to include source file paths (e.g., `filename.md`, `line 10`), providing transparency and allowing the user to verify the answer against the original documentation.

**Next Steps:** With the RAG pipeline functional and the UI complete, the final steps involve optimizing, hardening the code, and managing the overall project structure.

### Day 6: Code Hardening and Project Management

**Objective:** Day 6 was dedicated to refining the project structure, hardening the code against failures, and ensuring a smooth setup process for future users.

**What I Learned & Accomplished:**

* **Robust Network Handling (Retries):** Based on transient network errors experienced during data ingestion, I implemented a robust `requests_retry_session` using `urllib3.util.retry`. This logic ensures that the script automatically retries the GitHub download a specified number of times when encountering temporary connection issues (like `IncompleteRead`), dramatically improving reliability.
* **Environment Management with `uv`:** I standardized the environment management using `uv`, which is significantly faster than traditional `pip` and `venv`. I updated the `README.md` to clearly instruct users on using `uv venv` and `uv pip install -r requirements.txt`.
* **Testing and Debugging:** I introduced explicit error handling (e.g., `try...except` blocks) around critical I/O and API calls to catch failures gracefully, printing helpful messages instead of crashing the application.
* **Updated README:** I finalized the detailed installation and usage instructions in the `README.md` to cover all three operational modes: `uv run python ingest.py`, `uv run python main.py` (CLI test), and `uv run streamlit run app.py`.

**Next Steps:** The final day will focus on final project documentation, review, and preparing the repository for its final submission.

### Day 7: Final Review, Documentation, and Submission

**Objective:** The final day focused on ensuring the project meets all requirements, documenting the journey, and finalizing the repository for submission.

**What I Learned & Accomplished:**

* **Comprehensive Documentation:** I completed the `README.md` to include the full 7-day learning log, detailed installation instructions, and clear usage examples for all scripts.
* **Code Cleanup and Final Review:** I performed a final pass on all Python files, ensuring consistent formatting, removing unused imports, and adding clear comments where necessary.
* **Dependency Management:** I ensured the `requirements.txt` file was perfectly synchronized with the packages used in the project, reflecting only the necessary dependencies for the final, clean build.
* **Project Reflection:** The entire process from data ingestion and LLM enrichment to hybrid indexing, agent creation, and web deployment provided a complete, practical understanding of modern RAG system development and the powerful role of open-source tools in the AI ecosystem.