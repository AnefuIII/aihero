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