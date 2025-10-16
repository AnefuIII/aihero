import numpy as np
import io
from pypdf import PdfReader # <-- NEW: Import PdfReader

from minsearch import Index, VectorSearch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm 

# --- NEW: Function to read a local PDF file ---
def read_pdf_data(pdf_path):
    """Reads all text from a local PDF file and returns it as a single 'document'."""
    print(f"Reading PDF from: {pdf_path}")
    reader = PdfReader(pdf_path)
    all_text = ""
    for page in reader.pages:
        all_text += page.extract_text() + "\n\n"
    
    # Return data as a list containing a single 'document'
    # The 'filename' is the path for easy reference.
    return [{
        'filename': pdf_path,
        'content': all_text
    }]

# --- Existing Functions (Sliding Window, Chunk) ---

def sliding_window(seq, size, step):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        batch = seq[i:i+size]
        result.append({'start': i, 'content': batch})
        if i + size >= n:
            break

    return result


def chunk_documents(docs, size=2000, step=1000):
    chunks = []

    for doc in docs:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop('content')
        doc_chunks = sliding_window(doc_content, size=size, step=step)
        for chunk in doc_chunks:
            chunk.update(doc_copy)
        chunks.extend(doc_chunks)

    return chunks

# --- MODIFIED HYBRID INDEXING FUNCTION ---

# The repo_owner/repo_name arguments are no longer needed.
# It now accepts a 'data_source' argument, which should be the PDF path.
def index_all_data(
    data_source, # <-- CHANGED: Accepts the local PDF file path
    filter=None,
    chunking_params=None,
    embedding_model_name='multi-qa-distilbert-cos-v1'
):
    # 1. Read Data - Using the new PDF function
    docs = read_pdf_data(data_source) # <-- MODIFIED

    # 2. Filter Data (if a filter function is provided)
    # The filter function here might be less relevant for a single PDF, 
    # but the logic remains for future use.
    if filter is not None:
        docs = [doc for doc in docs if filter(doc)]

    # 3. Chunk Data
    if chunking_params is None:
        chunking_params = {'size': 2000, 'step': 1000}
        
    chunks = chunk_documents(docs, **chunking_params)
    
    # 4. Initialize Embedding Model
    print("Loading embedding model...")
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # 5. Generate Embeddings for Vector Index
    print(f"Generating embeddings for {len(chunks)} chunks...")
    chunk_embeddings = []
    for chunk in tqdm(chunks):
        v = embedding_model.encode(chunk['content'])
        chunk_embeddings.append(v)
    
    chunk_embeddings = np.array(chunk_embeddings)

    # 6. Create Text Index (minsearch Index)
    print("Building Text Index...")
    aut_index = Index(
        text_fields=["content", "filename"],
        keyword_fields=[]
    )
    aut_index.fit(chunks)

    # 7. Create Vector Index (minsearch VectorSearch)
    print("Building Vector Index...")
    autogen_vindex = VectorSearch()
    autogen_vindex.fit(chunk_embeddings, chunks)

    # 8. Return all three required components for Hybrid Search
    return aut_index, autogen_vindex, embedding_model