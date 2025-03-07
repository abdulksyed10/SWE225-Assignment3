import json
import time
import nltk
from nltk.stem import PorterStemmer

# Define the index file
MERGED_INDEX_FILE = "final_inverted_index.json"

# Initialize NLTK
nltk.download('punkt')
stemmer = PorterStemmer()

with open(MERGED_INDEX_FILE, "r", encoding="utf-8") as f:
    inverted_index = json.load(f)

def preprocess_query(query):
    """ Tokenizes and stems the query for better matching. """
    tokens = nltk.word_tokenize(query.lower())  # Convert to lowercase & tokenize
    return [stemmer.stem(token) for token in tokens if token.isalnum()]  # Remove non-alphanumeric

def search(query):
    """ Searches the indexed data and returns ranked results based on TF-IDF. """
    
    start_time = time.time()  # Start measuring response time

    query_terms = preprocess_query(query)
    results = {}

    # Retrieve relevant documents for query terms
    for term in query_terms:
        if term in inverted_index:
            for doc, score in inverted_index[term].items():
                # 🚨 Removed anchor text and heading boost logic
                results[doc] = results.get(doc, 0) + score  # Just use TF-IDF

    # Rank results by TF-IDF score (higher is better)
    ranked_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    end_time = time.time()  # End response time measurement
    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds

    # Display results
    if ranked_results:
        print(f"\n🔎 Search Results for: \"{query}\"")
        for i, (doc, score) in enumerate(ranked_results[:10], 1):
            print(f"{i}. {doc} (Score: {score:.4f})")

        print(f"\n⏱ Search completed in {elapsed_time:.2f} ms")
    else:
        print(f"\n❌ No results found for \"{query}\".")

if __name__ == "__main__":
    while True:
        query = input("\nEnter search query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        search(query)
