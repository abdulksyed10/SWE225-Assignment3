import json
import nltk
from nltk.stem import PorterStemmer

# Load the inverted index
INDEX_FILE = "inverted_index.json"

with open(INDEX_FILE, "r", encoding="utf-8") as f:
    inverted_index = json.load(f)

# Initialize stemmer
stemmer = PorterStemmer()

def preprocess_query(query):
    tokens = nltk.word_tokenize(query.lower())
    return [stemmer.stem(token) for token in tokens if token.isalnum()]

def search(query):
    query_terms = preprocess_query(query)
    results = {}

    for term in query_terms:
        if term in inverted_index:
            for doc, score in inverted_index[term].items():
                results[doc] = results.get(doc, 0) + score

    # Rank results by TF-IDF score
    ranked_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    # Display top results
    if ranked_results:
        print("\nTop Search Results:")
        for i, (doc, score) in enumerate(ranked_results[:10], 1):
            print(f"{i}. {doc} (Score: {score:.4f})")
    else:
        print("\nNo results found.")

if __name__ == "__main__":
    while True:
        query = input("\nEnter search query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        search(query)
