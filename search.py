import json
import time
import nltk
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Define the index file
MERGED_INDEX_FILE = "final_inverted_index.json"

# Initialize NLTK
nltk.download('punkt')
nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Load the inverted index with error handling
try:
    with open(MERGED_INDEX_FILE, "r", encoding="utf-8") as f:
        inverted_index = json.load(f)
except FileNotFoundError:
    print(f"Error: The index file '{MERGED_INDEX_FILE}' was not found.")
    inverted_index = {}

def preprocess_query(query):
    #Tokenizes, stems, and removes stopwords from the query. 
    tokens = nltk.word_tokenize(query.lower())  # Convert to lowercase & tokenize
    return [
        stemmer.stem(token) for token in tokens 
        if token.isalnum() and token not in stop_words  # Remove stopwords & non-alphanumeric
    ]

def normalize_scores(results):
    #Normalizes TF-IDF scores between 0 and 1 for fair ranking. 
    if not results:
        return results
    
    max_score = max(results.values())
    return {doc: score / max_score for doc, score in results.items()}

def search(query):
    #Searches the indexed data and returns ranked results based on normalized TF-IDF. 
    
    if not inverted_index:
        print("\n The index is empty. Please ensure the index file is correctly generated.")
        return

    start_time = time.time()  # Start measuring response time

    query_terms = preprocess_query(query)
    results = {}

    # Retrieve relevant documents for query terms
    for term in query_terms:
        if term in inverted_index:
            for doc, score in inverted_index[term].items():
                results[doc] = results.get(doc, 0) + score  # Accumulate TF-IDF scores

    # Normalize scores for fair ranking
    results = normalize_scores(results)

    # Rank results by TF-IDF score (higher is better)
    ranked_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    end_time = time.time()  # End response time measurement
    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds

    # Display results
    if ranked_results:
        print(f"\n Search Results for: \"{query}\"")
        for i, (doc, score) in enumerate(ranked_results[:10], 1):
            print(f"{i}. {doc} (Score: {score:.4f})")

        print(f"\n Search completed in {elapsed_time:.2f} ms")
    else:
        print(f"\n No results found for \"{query}\".")

if __name__ == "__main__":
    while True:
        query = input("\nEnter search query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        search(query)
