import os
import json
import math
import hashlib
import nltk
from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.stem import PorterStemmer
from tqdm import tqdm
from urllib.parse import urlparse, urlunparse

nltk.download('punkt')

# Initialize stemmer
stemmer = PorterStemmer()

DATA_DIR = "DEV"
INDEX_DIR = "partial_indexes"
MERGED_INDEX_FILE = "final_inverted_index.json"
DOCS_PER_PARTIAL_INDEX = 1000  # Save index to disk every 1000 docs

# Ensure index directory exists
os.makedirs(INDEX_DIR, exist_ok=True)

# Track processed document hashes
processed_hashes = set()

# Function to normalize URLs
def normalize_url(url):
    parsed = urlparse(url)
    normalized_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    return normalized_url.replace("http://", "https://")  # Force HTTPS

# Function to compute SHA-256 hash of text
def compute_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [stemmer.stem(token) for token in tokens if token.isalnum()]
    return tokens

# Function to save partial index to disk
def save_partial_index(partial_index, index_num):
    partial_index_file = os.path.join(INDEX_DIR, f"partial_index_{index_num}.json")
    with open(partial_index_file, "w", encoding="utf-8") as f:
        json.dump(partial_index, f, indent=4)
    print(f"Saved partial index: {partial_index_file}")

# Function to merge all partial indexes into one final inverted index
def merge_partial_indexes():
    final_index = defaultdict(lambda: defaultdict(int))

    print("\nMerging partial indexes...")

    for file in sorted(os.listdir(INDEX_DIR)):
        if file.startswith("partial_index_") and file.endswith(".json"):
            file_path = os.path.join(INDEX_DIR, file)
            with open(file_path, "r", encoding="utf-8") as f:
                partial_index = json.load(f)
                for term, postings in partial_index.items():
                    for doc_id, tf in postings.items():
                        final_index[term][doc_id] += tf  # Merge TF counts

    # Compute TF-IDF scores after merging
    print("Computing TF-IDF scores...")
    doc_count = len(set(doc for postings in final_index.values() for doc in postings))
    for term, doc_dict in final_index.items():
        for doc_id, tf in doc_dict.items():
            idf = math.log(doc_count / (1 + len(doc_dict)))
            final_index[term][doc_id] = tf * idf  # Replace TF with TF-IDF

    # Save final merged index
    with open(MERGED_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(final_index, f, indent=4)

    print(f"Final merged index saved as {MERGED_INDEX_FILE}")

def build_index():
    partial_index_counter = 0
    doc_count = 0
    doc_freq = defaultdict(int)
    inverted_index = defaultdict(lambda: defaultdict(int))

    # Count total documents
    total_docs = sum(len(files) for _, _, files in os.walk(DATA_DIR))

    print(f"Starting indexing for {total_docs} documents...")

    with tqdm(total=total_docs, desc="Indexing Progress", unit="doc") as pbar:
        for folder in os.listdir(DATA_DIR):
            folder_path = os.path.join(DATA_DIR, folder)
            if not os.path.isdir(folder_path):
                continue

            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if not file.endswith(".json"):
                    continue
                
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    url = normalize_url(data.get("url", file))  
                    print(f"📝 Extracted URL from JSON: {data.get('url', file)} (From file: {file_path})")
                    html_content = data.get("content", "")

                    # Parse HTML and extract text
                    soup = BeautifulSoup(html_content, "html.parser")
                    title = soup.title.string if soup.title else ""
                    headings = " ".join([h.text for h in soup.find_all(["h1", "h2", "h3"])])
                    bold_text = " ".join([b.text for b in soup.find_all(["b", "strong"])])
                    body_text = soup.get_text()

                    full_text = f"{title} {headings} {bold_text} {body_text}"

                    # Compute content hash to detect duplicates
                    doc_hash = compute_hash(full_text)

                    if doc_hash in processed_hashes:
                        print(f"Skipping duplicate page: {url}")
                        continue  

                    processed_hashes.add(doc_hash)

                    tokens = preprocess_text(full_text)
                    doc_count += 1
                    unique_terms = set()

                    # ✅ Removed Anchor Text Indexing Here

                    for token in tokens:
                        inverted_index[token][url] += 1
                        unique_terms.add(token)

                    for term in unique_terms:
                        doc_freq[term] += 1

                    # Offload partial index to disk every 1000 documents
                    if doc_count % DOCS_PER_PARTIAL_INDEX == 0:
                        save_partial_index(inverted_index, partial_index_counter)
                        inverted_index.clear()  # Clear memory
                        partial_index_counter += 1

                    pbar.update(1)

    # Save remaining index if any documents left
    if inverted_index:
        save_partial_index(inverted_index, partial_index_counter)

    print("\nIndexing completed!")
    print(f"Indexed {doc_count} unique documents.")
    print(f"Saved {partial_index_counter + 1} partial index files.")

    # Merge partial indexes into the final index
    merge_partial_indexes()

if __name__ == "__main__":
    build_index()