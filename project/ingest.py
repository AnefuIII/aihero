import io
import zipfile
import requests
import frontmatter
import numpy as np

from minsearch import Index, VectorSearch # <-- New Import
from sentence_transformers import SentenceTransformer # <-- New Import
from tqdm.auto import tqdm # Import tqdm for progress bar (optional, but helpful)


# --- Existing Functions (Read, Sliding Window, Chunk) ---

def read_repo_data(repo_owner, repo_name):
    url = f'https://codeload.github.com/{repo_owner}/{repo_name}/zip/refs/heads/main'
    resp = requests.get(url)

    repository_data = []

    zf = zipfile.ZipFile(io.BytesIO(resp.content))

    for file_info in zf.infolist():
        filename = file_info.filename.lower()

        if not (filename.endswith('.md') or filename.endswith('.mdx')):
            continue

        with zf.open(file_info) as f_in:
            content = f_in.read()
            post = frontmatter.loads(content)
            data = post.to_dict()

            _, filename_repo = file_info.filename.split('/', maxsplit=1)
            data['filename'] = filename_repo
            repository_data.append(data)

    zf.close()

    return repository_data


def sliding_window(seq, size, step):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(seq)
    result = []
    for i in range(0, n, step):
        batch = seq[i:i+size]
        result.append({'start': i, 'content': batch})
        if i + size >= n: # Changed > to >= for cleaner end condition
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

# --- NEW HYBRID INDEXING FUNCTION ---

# This function is the new entry point for main.py
def index_all_data(
    repo_owner,
    repo_name,
    filter=None,
    chunking_params=None,
    embedding_model_name='multi-qa-distilbert-cos-v1'
):
    # 1. Read Data
    docs = read_repo_data(repo_owner, repo_name)

    # 2. Filter Data (if a filter function is provided)
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



# --- OLD INDEXING FUNCTION (FOR REFERENCE) ---

# import io
# import zipfile
# import requests
# import frontmatter

# from minsearch import Index


# def read_repo_data(repo_owner, repo_name):
#     url = f'https://codeload.github.com/{repo_owner}/{repo_name}/zip/refs/heads/main'
#     #url = f'https://github.com/{repo_owner}/{repo_name}/archive/refs/heads/main.zip'
#     resp = requests.get(url)

#     repository_data = []

#     zf = zipfile.ZipFile(io.BytesIO(resp.content))

#     for file_info in zf.infolist():
#         filename = file_info.filename.lower()

#         if not (filename.endswith('.md') or filename.endswith('.mdx')):
#             continue

#         with zf.open(file_info) as f_in:
#             content = f_in.read()
#             post = frontmatter.loads(content)
#             data = post.to_dict()

#             _, filename_repo = file_info.filename.split('/', maxsplit=1)
#             data['filename'] = filename_repo
#             repository_data.append(data)

#     zf.close()

#     return repository_data


# def sliding_window(seq, size, step):
#     if size <= 0 or step <= 0:
#         raise ValueError("size and step must be positive")

#     n = len(seq)
#     result = []
#     for i in range(0, n, step):
#         batch = seq[i:i+size]
#         result.append({'start': i, 'content': batch})
#         if i + size > n:
#             break

#     return result


# def chunk_documents(docs, size=2000, step=1000):
#     chunks = []

#     for doc in docs:
#         doc_copy = doc.copy()
#         doc_content = doc_copy.pop('content')
#         doc_chunks = sliding_window(doc_content, size=size, step=step)
#         for chunk in doc_chunks:
#             chunk.update(doc_copy)
#         chunks.extend(doc_chunks)

#     return chunks


# def index_data(
#         repo_owner,
#         repo_name,
#         filter=None,
#         chunk=False,
#         chunking_params=None,
#     ):
#     docs = read_repo_data(repo_owner, repo_name)

#     if filter is not None:
#         docs = [doc for doc in docs if filter(doc)]

#     if chunk:
#         if chunking_params is None:
#             chunking_params = {'size': 2000, 'step': 1000}
#         docs = chunk_documents(docs, **chunking_params)

#     index = Index(
#         text_fields=["content", "filename"],
#     )

#     index.fit(docs)
#     return index