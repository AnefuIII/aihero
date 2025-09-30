# 🤖 AutoGen Documentation AI Assistant

**A Retrieval-Augmented Generation (RAG) agent that uses Hybrid Search to answer technical questions based on the official Microsoft AutoGen documentation.**

---

## 2. Overview

### The Problem

Navigating large, complex, and rapidly evolving open-source project documentation, such as the AutoGen framework, can be time-consuming. Developers often need quick, accurate answers grounded in the latest source files, not generic web search results.

### The Solution

This project implements a sophisticated AI Agent powered by **Hybrid Search (Sparse & Dense Retrieval)** to provide instant, precise, and cited answers directly from the AutoGen documentation. This allows users to quickly find configuration details, API usage, and conceptual explanations without manually sifting through hundreds of files.

### Key Features

* **Hybrid Search:** Combines traditional keyword search with vector similarity search for superior retrieval accuracy.
* **Grounded Answers:** Answers are strictly based on the content of the repository files, minimizing hallucinations.
* **Live Web App:** Deployed via Streamlit for immediate, cross-platform access.

<p align="center">
  <img src="doc_files/Screenshot%20(195).png" alt="Screenshot of the Streamlit App" width="600"/>
</p>

---

## 3. Installation

### Prerequisites

To set up and run this project locally, you will need:

* Python 3.9+
* Git
* An **OpenAI API Key** (required for the underlying LLM calls, set as an environment variable or in a `.env` file).

### Step-by-Step Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AnefuIII/aihero/](https://github.com/AnefuIII/aihero/)
    cd aihero
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set your API Key:**
    The agent expects the key to be set as an environment variable.
    ```bash
    export OPENAI_API_KEY="YOUR_API_KEY"
    ```

---

## 4. Usage

### Running the Data Ingestion (Indexing)

The first step is to run the data ingestion to pull the documents, chunk them, and create the hybrid indexes. This only needs to be run once.

```bash
# This command runs the indexing logic found in ingest.py
python ingest.py
```

### Running the Streamlit Web Application
Start the interactive chat interface locally:

```Bash

streamlit run app.py
```
The application will open in your web browser, ready to answer questions about the AutoGen documentation.

## 5. Features
#### Core Agent Capabilities
| Feature | Description |
|---------|-------------|
| **Hybrid RAG** | Utilizes both vector and keyword search (as defined in `search_agent.py`) to maximize retrieval quality. |
| **Selective Indexing** | Automatically filters out non-documentation files (e.g., `examples/`) during ingestion for cleaner results. |
| **Chat History** | Maintains conversation history within the Streamlit session state. |
| **Interaction Logging** | Logs user prompts and agent responses to a local file (`logs.py`) for later analysis. |

#### Roadmap
* [ ] Implement fine-grained access control for private repository documentation.

* [ ] Introduce a multi-agent system for complex query decomposition.

* [ ] Add a feature to visualize the source chunks retrieved by the agent.

## 6. Contributing
Guidelines for contributions are appreciated! Please fork the repository and submit a pull request with your suggested changes.

## 7. Tests and Evaluation
#### Agent Evaluation Metrics
The agent's performance is monitored using a set of evaluation criteria applied to synthetic questions. Our evaluation notebook (eval/Day_5_Evaluation_Assignment.ipynb) assesses the following:

| Metric | Your Current Score | Description | Importance |
|--------|-------------------|-------------|------------|
| `instructions_follow` | 81.82% | The agent followed all explicit user instructions. | High |
| `instructions_avoid` | 100.00% | The agent avoided doing things it was explicitly told not to do. | High |
| `answer_relevant` | 100.00% | The response directly addresses the user's question. | **CRITICAL** |
| `answer_clear` | 100.00% | The final answer is clear, concise, and accurate. | High |
| `answer_citations` | 81.82% | The response includes proper citations or sources (links) when required. | High |
| `completeness` | 100.00% | The response is complete and covers all key aspects of the request. | Medium |
| `tool_call_search` | 54.55% | The agent correctly invoked the search tool when necessary. | Medium |


#### Summary & Next Steps
* The most critical metric, answer_relevant, is currently at 100%, which is excellent.

* Key areas for immediate improvement are tool_call_search (54.55%) and instructions\_follow (81.82%). Improving the tool-calling reliability will likely boost the citation score as well.

* Running Tests: The full evaluation suite can be executed using the Jupyter notebook:

```Bash

jupyter lab eval/Day_5_Evaluation_Assignment.ipynb
```
## 8. Deployment

This application is currently deployed on the Streamlit Community Cloud for public access.

**Live App:** [AutoGen Documentation Assistant](https://aihero-fcjrcex64na3e7z53r2eh5.streamlit.app/)

#### Deployment Steps (Streamlit Cloud)
1. Ensure a requirements.txt file exists with all dependencies.

2. Ensure your OPENAI_API_KEY is set as a Secret in the Streamlit Cloud dashboard.

3. Deploy the app.py file directly from the GitHub repository branch.

## 9. FAQ / Troubleshooting
Q: Why do I get an Index Error during ingestion?
A: Ensure your local environment has sufficient memory to handle the embedding process for the entire repository. Try running ingestion on a machine with more RAM.

Q: My agent is running but not providing complete answers.
A: Check your OPENAI_API_KEY to ensure it is valid and that your account has sufficient quota for the model being used (e.g., gpt-4o-mini).

## 10. Credits / Acknowledgments

This project was built as part of the AI Hero course, guided by Alexey Grigorev.

**Course:** [AI Hero Course by Alexey Grigorev](https://alexeygrigorev.com/aihero/)

Inspiration: The architecture is based on the principles of advanced RAG systems.

## 11. License
This project is licensed under the [LICENSE TYPE HERE] License.

(Note: Replace [LICENSE TYPE HERE] with your chosen license, e.g., MIT, Apache 2.0.)