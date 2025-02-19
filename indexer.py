import os
import re
import json
import math
import nltk
from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.stem import PorterStemmer
from tqdm import tqdm  # Progress bar

nltk.download('punkt')

# Initialize stemmer
stemmer = PorterStemmer()

# Directory containing HTML files
DATA_DIR = "DEV"

# Index file path
INDEX_FILE = "inverted_index.json"

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())  # Lowercase and tokenize
    tokens = [stemmer.stem(token) for token in tokens if token.isalnum()]  # Stemming & removing non-alphanumeric
    return tokens

def build_index():
    inverted_index = defaultdict(lambda: defaultdict(int))
    doc_freq = defaultdict(int)
    doc_count = 0

    # Count total documents for progress tracking
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
                    url = data.get("url", file)
                    html_content = data.get("content", "")

                    # Parse HTML and extract text
                    soup = BeautifulSoup(html_content, "html.parser")
                    title = soup.title.string if soup.title else ""
                    headings = " ".join([h.text for h in soup.find_all(["h1", "h2", "h3"])])
                    bold_text = " ".join([b.text for b in soup.find_all(["b", "strong"])])
                    body_text = soup.get_text()

                    full_text = f"{title} {headings} {bold_text} {body_text}"
                    tokens = preprocess_text(full_text)

                    doc_count += 1
                    unique_terms = set()

                    for token in tokens:
                        inverted_index[token][url] += 1
                        unique_terms.add(token)

                    for term in unique_terms:
                        doc_freq[term] += 1

                    pbar.update(1)  # Update progress bar

    # Compute TF-IDF
    print("Computing TF-IDF scores...")
    for term, doc_dict in inverted_index.items():
        for doc_id, tf in doc_dict.items():
            idf = math.log(doc_count / (1 + doc_freq[term]))
            inverted_index[term][doc_id] = tf * idf  # Replace TF with TF-IDF

    # Save to file
    print("Saving index to disk...")
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(inverted_index, f, indent=4)

    # Final Summary
    print(f"\nIndexing completed!")
    print(f"Indexed {doc_count} documents.")
    print(f"Unique tokens: {len(inverted_index)}")
    print(f"Index size: {os.path.getsize(INDEX_FILE) / 1024:.2f} KB")

if __name__ == "__main__":
    build_index()
