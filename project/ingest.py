import io
import os
import time
import json
import pickle
import requests
import numpy as np
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from pypdf import PdfReader
from tqdm.auto import tqdm

from minsearch import Index, VectorSearch
from sentence_transformers import SentenceTransformer

# Playwright imports
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# ---------------------------
# PDF helpers
# ---------------------------
def read_pdf_data(pdf_source):
    """Read PDF text and return list of document dicts."""
    try:
        reader = PdfReader(pdf_source)
    except Exception as e:
        print(f"[PDF] Error opening PDF source: {e}")
        return []

    all_text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text:
            all_text += page_text + "\n--- PAGE BREAK ---\n"

    if not all_text.strip():
        return []

    return [{"filename": str(pdf_source), "content": all_text.strip()}]

def download_and_read_pdf(url: str):
    """Download PDF into memory and extract text."""
    print(f"[PDF] Downloading PDF: {url}")
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        pdf_bytes = io.BytesIO(resp.content)
        docs = read_pdf_data(pdf_bytes)
        if docs:
            docs[0]["filename"] = "National Housing Fund (NHF) Act [PDF]"
            docs[0]["url"] = url
        return docs
    except requests.RequestException as e:
        print(f"[PDF] Download error: {e}")
        return []

# ---------------------------
# Playwright dynamic page scraper
# ---------------------------
def scrape_dynamic_page(url: str, base_url: str, headless=True):
    """Render a page, expand accordions, return document dict."""
    print(f"[SCRAPE] Rendering: {url}")
    rendered_html = None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=30000)
        except PlaywrightTimeoutError:
            page.goto(url, wait_until="load", timeout=30000)

        # Scroll to trigger lazy load
        try:
            page.evaluate(
                """() => {
                    return new Promise(resolve => {
                        let distance = 400;
                        let total = 0;
                        let timer = setInterval(() => {
                            window.scrollBy(0, distance);
                            total += distance;
                            if (total > document.body.scrollHeight) {
                                clearInterval(timer);
                                resolve(true);
                            }
                        }, 150);
                    });
                }"""
            )
        except Exception:
            pass

        # Click expanders
        selectors = ["button[aria-expanded]", ".accordion-button", ".read-more", ".show-more"]
        for sel in selectors:
            try:
                loc = page.locator(sel)
                for i in range(loc.count()):
                    try:
                        loc.nth(i).click(timeout=200)
                        time.sleep(0.05)
                    except Exception:
                        pass
            except Exception:
                pass

        time.sleep(0.3)
        try:
            rendered_html = page.content()
        except Exception:
            pass

        browser.close()

    if not rendered_html:
        return None

    soup = BeautifulSoup(rendered_html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title_tag = soup.find("title")
    page_title = title_tag.text.strip() if title_tag else url.split("/")[-1]
    filename = f"{page_title.replace(' | FMBN', '').strip()} - FMBN Website"
    text_content = soup.get_text(separator="\n", strip=True)

    links = set()
    base_netloc = urlparse(base_url).netloc
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("javascript:") or href.startswith("#"):
            continue
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if parsed.netloc == base_netloc and not full.lower().endswith((".png", ".jpg", ".gif", ".zip", ".pdf")):
            links.add(full.rstrip("/"))

    return {"filename": filename, "content": text_content, "url": url, "links": list(links)}

def scrape_website_dynamic(base_url: str, pdf_url: str = None, max_pages: int = 40, headless: bool = True):
    """Breadth-first crawl starting from base_url using rendered pages."""
    base_url = base_url.rstrip("/")
    to_visit = [base_url]
    visited = set()
    docs = []

    print(f"[CRAWL] Starting crawl: {base_url} (max_pages={max_pages})")

    while to_visit and len(visited) < max_pages:
        current = to_visit.pop(0)
        if current in visited:
            continue

        try:
            doc = scrape_dynamic_page(current, base_url, headless=headless)
        except Exception as e:
            print(f"[CRAWL] Error scraping {current}: {e}")
            doc = None

        visited.add(current)
        if doc:
            docs.append({"filename": doc["filename"], "content": doc["content"], "url": doc["url"]})
            for link in doc.get("links", []):
                if link not in visited and link not in to_visit:
                    to_visit.append(link)

    if pdf_url:
        pdf_docs = download_and_read_pdf(pdf_url)
        if pdf_docs:
            docs.extend(pdf_docs)

    print(f"[CRAWL] Finished. Pages/Documents ingested: {len(docs)}")
    return docs

# ---------------------------
# Chunking & persistence
# ---------------------------
def sliding_window(seq: str, size: int = 2000, step: int = 1000):
    result = []
    n = len(seq)
    for i in range(0, n, step):
        batch = seq[i:i + size]
        result.append({"start": i, "content": batch})
        if i + size >= n:
            break
    return result

def chunk_documents(docs, size: int = 2000, step: int = 1000):
    chunks = []
    for doc in docs:
        text = doc.pop("content", "")
        if not text:
            continue
        for ck in sliding_window(text, size=size, step=step):
            ck.update(doc.copy())
            chunks.append(ck)
    return chunks

# ---------------------------
# Production-ready indexing
# ---------------------------
def index_website_data(
    website_url,
    pdf_url=None,
    chunk_file="chunks.pkl",
    emb_file="embeddings.npy",
    embedding_model_name="multi-qa-distilbert-cos-v1",
    max_pages=40,
    headless=True
):
    """Load persisted chunks/embeddings or crawl/build them."""
    if os.path.exists(chunk_file) and os.path.exists(emb_file):
        print("[INDEX] Loading persisted chunks and embeddings from disk...")
        with open(chunk_file, "rb") as f:
            chunks = pickle.load(f)
        chunk_embeddings = np.load(emb_file)
    else:
        # Crawl + PDF ingestion
        docs = scrape_website_dynamic(website_url, pdf_url, max_pages=max_pages, headless=headless)
        chunks = chunk_documents(docs)
        # Embedding
        print("[INDEX] Loading embedding model...")
        embedding_model = SentenceTransformer(embedding_model_name)
        print(f"[INDEX] Generating embeddings for {len(chunks)} chunks...")
        chunk_embeddings = np.array([embedding_model.encode(c["content"]) for c in tqdm(chunks)])
        # Persist to disk
        with open(chunk_file, "wb") as f:
            pickle.dump(chunks, f)
        np.save(emb_file, chunk_embeddings)
        print("[INDEX] Chunks and embeddings saved to disk.")

    # Build minsearch indices
    print("[INDEX] Building text index...")
    aut_index = Index(text_fields=["content", "filename"], keyword_fields=[])
    aut_index.fit(chunks)
    print("[INDEX] Building vector index...")
    autogen_vindex = VectorSearch()
    autogen_vindex.fit(chunk_embeddings, chunks)

    # Load embedding model if not already loaded
    if 'embedding_model' not in locals():
        embedding_model = SentenceTransformer(embedding_model_name)

    print("[INDEX] Indexing complete.")
    return aut_index, autogen_vindex, embedding_model


# import numpy as np
# import io
# import requests # <-- NEW: For HTTP requests (scraping & downloading)
# from bs4 import BeautifulSoup # <-- NEW: For parsing HTML
# from pypdf import PdfReader
# import tempfile # <-- NEW: To handle temporary files for PDF download

# from minsearch import Index, VectorSearch
# from sentence_transformers import SentenceTransformer
# from tqdm.auto import tqdm 

# # --- Existing: Function to read a local PDF file (Keep for structure, but modify for memory usage) ---
# def read_pdf_data(pdf_source):
#     """
#     Reads all text from a PDF source (file path or file-like object) 
#     and returns it as a single 'document'.
#     """
#     print(f"Reading PDF from: {pdf_source}")
#     # If pdf_source is a path/filename, PdfReader handles it.
#     # If it's a file-like object (like io.BytesIO), PdfReader handles it.
#     reader = PdfReader(pdf_source) 
#     all_text = ""
#     for page in reader.pages:
#         all_text += page.extract_text() + "\n\n"
    
#     # Return data as a list containing a single 'document'
#     # Source is captured in the filename field.
#     return [{
#         'filename': str(pdf_source), # Use source name/path/URL
#         'content': all_text
#     }]

# # --- NEW: Function to download and read a PDF from a URL ---
# def download_and_read_pdf(url: str):
#     """Downloads a PDF from a URL and reads its content using a temporary file."""
#     print(f"Downloading and processing PDF from: {url}")
    
#     try:
#         response = requests.get(url, stream=True, timeout=30)
#         response.raise_for_status() # Raise an exception for bad status codes
        
#         # Use io.BytesIO to read the PDF directly from memory without saving to disk
#         pdf_bytes = io.BytesIO(response.content)
        
#         # Now use the existing PDF reading logic
#         doc_list = read_pdf_data(pdf_bytes)
        
#         # Correct the 'filename' to the URL for proper citation
#         if doc_list:
#             doc_list[0]['filename'] = url
        
#         return doc_list
        
#     except requests.RequestException as e:
#         print(f"Error downloading PDF from {url}: {e}")
#         return []
        
# # --- NEW: Function to scrape a specific URL and title the document ---
# def scrape_url_to_document(url: str, base_url: str):
#     """Scrapes a single URL, uses the page <title> as the source name."""
#     print(f"Scraping URL: {url}")
    
#     try:
#         response = requests.get(url, timeout=15)
#         response.raise_for_status()
#         soup = BeautifulSoup(response.content, 'html.parser')
        
#         # KEY IMPROVEMENT: Use the HTML <title> tag for the filename (source)
#         # This provides a relevant, human-readable source title for every chunk.
#         title_tag = soup.find('title')
#         page_title = title_tag.text.strip() if title_tag else url.split('/')[-1]
        
#         # Use a cleaner version of the title for the filename
#         filename_source = f"{page_title} - FMBN Website"
        
#         # Extract all visible text
#         # You might want to focus on the main content area (e.g., a div with id="main-content") 
#         # for cleaner RAG, but get_text() is the simplest general approach.
#         text_content = soup.get_text(separator=' ', strip=True)
        
#         # Get all internal links on this page (used for crawling)
#         links = set()
#         for a_tag in soup.find_all('a', href=True):
#             href = a_tag['href']
#             # Only consider links that start with the base URL or relative paths
#             if href.startswith(base_url) or href.startswith('/'):
#                 # Convert relative links to absolute links
#                 full_url = requests.compat.urljoin(base_url, href)
#                 # Ignore fragment identifiers (#) and common external files
#                 if '#' not in full_url and not full_url.endswith(('.png', '.jpg', '.gif', '.zip')):
#                     # Only add links within the same domain path for a simple crawler
#                     if full_url.startswith(base_url):
#                         links.add(full_url.rstrip('/'))

#         return {
#             'filename': filename_source, 
#             'content': text_content,
#             'url': url,
#             'links': list(links) # Return links for the crawler
#         }
#     except requests.RequestException as e:
#         print(f"Error scraping {url}: {e}")
#         return None

# def scrape_website(base_url, pdf_url, max_pages=40):
#     all_docs = []
#     urls_to_visit = {base_url.rstrip('/')}
#     urls_processed = set()

#     while urls_to_visit and len(urls_processed) < max_pages:
#         current_url = urls_to_visit.pop()

#         if current_url in urls_processed:
#             continue

#         doc_data = scrape_url_to_document(current_url, base_url)
#         urls_processed.add(current_url)

#         if doc_data:
#             all_docs.append({
#                 'filename': doc_data['filename'],
#                 'content': doc_data['content'],
#                 'url': doc_data['url']
#             })

#             # Add all valid links for recursive crawling
#             for link in doc_data['links']:
#                 if link not in urls_processed:
#                     urls_to_visit.add(link)

#     # Add PDF as last doc
#     all_docs.extend(download_and_read_pdf(pdf_url))
#     return all_docs

# # --- Existing Functions (Sliding Window, Chunk) ---
# # ... (Keep these functions unchanged)
# def sliding_window(seq, size, step):
# # ... (sliding_window content)
#     n = len(seq)
#     result = []
#     for i in range(0, n, step):
#         batch = seq[i:i+size]
#         result.append({'start': i, 'content': batch})
#         if i + size >= n:
#             break
#     return result

# def chunk_documents(docs, size=2000, step=1000):
# # ... (chunk_documents content)
#     chunks = []
#     for doc in docs:
#         doc_copy = doc.copy()
#         doc_content = doc_copy.pop('content')
#         doc_chunks = sliding_window(doc_content, size=size, step=step)
#         for chunk in doc_chunks:
#             chunk.update(doc_copy)
#         chunks.extend(doc_chunks)
#     return chunks

# # --- MODIFIED HYBRID INDEXING FUNCTION ---

# # This function must be renamed and updated to accept the website and PDF URLs.
# def index_website_data( # <-- RENAMED
#     website_url, # <-- NEW: Accepts the website base URL
#     pdf_url,     # <-- NEW: Accepts the PDF URL
#     filter=None,
#     chunking_params=None,
#     embedding_model_name='multi-qa-distilbert-cos-v1'
# ):
#     # 1. Read Data - Using the new scraping function
#     docs = scrape_website(website_url, pdf_url) # <-- MODIFIED

#     # 2. Filter Data 
#     if filter is not None:
#         docs = [doc for doc in docs if filter(doc)]
        
#     # Check if any documents were successfully ingested
#     if not docs:
#         raise RuntimeError("No data was successfully scraped or read. Check URLs and network connection.")

#     # 3. Chunk Data
#     if chunking_params is None:
#         chunking_params = {'size': 2000, 'step': 1000}
        
#     chunks = chunk_documents(docs, **chunking_params)
    
#     # 4. Initialize Embedding Model
#     print("Loading embedding model...")
#     embedding_model = SentenceTransformer(embedding_model_name)
    
#     # 5. Generate Embeddings for Vector Index
#     print(f"Generating embeddings for {len(chunks)} chunks...")
#     chunk_embeddings = []
#     for chunk in tqdm(chunks):
#         v = embedding_model.encode(chunk['content'])
#         chunk_embeddings.append(v)
    
#     chunk_embeddings = np.array(chunk_embeddings)

#     # 6. Create Text Index (minsearch Index)
#     print("Building Text Index...")
#     aut_index = Index(
#         text_fields=["content", "filename"],
#         keyword_fields=[]
#     )
#     aut_index.fit(chunks)

#     # 7. Create Vector Index (minsearch VectorSearch)
#     print("Building Vector Index...")
#     autogen_vindex = VectorSearch()
#     autogen_vindex.fit(chunk_embeddings, chunks)

#     # 8. Return all three required components for Hybrid Search
#     return aut_index, autogen_vindex, embedding_model

# # --- Clean up the original index_all_data if it's no longer needed, 
# # --- or keep it if you still use it for local PDF testing.




# #for local PDF file ingestion and hybrid indexing

# # import numpy as np
# # import io
# # from pypdf import PdfReader 

# # import requests # <-- NEW: For HTTP requests (scraping & downloading)
# # from bs4 import BeautifulSoup # <-- NEW: For parsing HTML
# # import tempfile

# # from minsearch import Index, VectorSearch
# # from sentence_transformers import SentenceTransformer
# # from tqdm.auto import tqdm 

# # # --- NEW: Function to read a local PDF file ---
# # def read_pdf_data(pdf_path):
# #     """Reads all text from a local PDF file and returns it as a single 'document'."""
# #     print(f"Reading PDF from: {pdf_path}")
# #     reader = PdfReader(pdf_path)
# #     all_text = ""
# #     for page in reader.pages:
# #         all_text += page.extract_text() + "\n\n"
    
# #     # Return data as a list containing a single 'document'
# #     # The 'filename' is the path for easy reference.
# #     return [{
# #         'filename': pdf_path,
# #         'content': all_text
# #     }]

# # # --- Existing Functions (Sliding Window, Chunk) ---

# # def sliding_window(seq, size, step):
# #     if size <= 0 or step <= 0:
# #         raise ValueError("size and step must be positive")

# #     n = len(seq)
# #     result = []
# #     for i in range(0, n, step):
# #         batch = seq[i:i+size]
# #         result.append({'start': i, 'content': batch})
# #         if i + size >= n:
# #             break

# #     return result


# # def chunk_documents(docs, size=2000, step=1000):
# #     chunks = []

# #     for doc in docs:
# #         doc_copy = doc.copy()
# #         doc_content = doc_copy.pop('content')
# #         doc_chunks = sliding_window(doc_content, size=size, step=step)
# #         for chunk in doc_chunks:
# #             chunk.update(doc_copy)
# #         chunks.extend(doc_chunks)

# #     return chunks

# # # --- MODIFIED HYBRID INDEXING FUNCTION ---

# # # The repo_owner/repo_name arguments are no longer needed.
# # # It now accepts a 'data_source' argument, which should be the PDF path.
# # def index_all_data(
# #     data_source, # <-- CHANGED: Accepts the local PDF file path
# #     filter=None,
# #     chunking_params=None,
# #     embedding_model_name='multi-qa-distilbert-cos-v1'
# # ):
# #     # 1. Read Data - Using the new PDF function
# #     docs = read_pdf_data(data_source) # <-- MODIFIED

# #     # 2. Filter Data (if a filter function is provided)
# #     # The filter function here might be less relevant for a single PDF, 
# #     # but the logic remains for future use.
# #     if filter is not None:
# #         docs = [doc for doc in docs if filter(doc)]

# #     # 3. Chunk Data
# #     if chunking_params is None:
# #         chunking_params = {'size': 2000, 'step': 1000}
        
# #     chunks = chunk_documents(docs, **chunking_params)
    
# #     # 4. Initialize Embedding Model
# #     print("Loading embedding model...")
# #     embedding_model = SentenceTransformer(embedding_model_name)
    
# #     # 5. Generate Embeddings for Vector Index
# #     print(f"Generating embeddings for {len(chunks)} chunks...")
# #     chunk_embeddings = []
# #     for chunk in tqdm(chunks):
# #         v = embedding_model.encode(chunk['content'])
# #         chunk_embeddings.append(v)
    
# #     chunk_embeddings = np.array(chunk_embeddings)

# #     # 6. Create Text Index (minsearch Index)
# #     print("Building Text Index...")
# #     aut_index = Index(
# #         text_fields=["content", "filename"],
# #         keyword_fields=[]
# #     )
# #     aut_index.fit(chunks)

# #     # 7. Create Vector Index (minsearch VectorSearch)
# #     print("Building Vector Index...")
# #     autogen_vindex = VectorSearch()
# #     autogen_vindex.fit(chunk_embeddings, chunks)

# #     # 8. Return all three required components for Hybrid Search
# #     return aut_index, autogen_vindex, embedding_model