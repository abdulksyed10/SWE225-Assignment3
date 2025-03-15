import json
import time
import nltk
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict
import math
from difflib import get_close_matches

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

# Tokenizes, stems, and removes stopwords from the query.
def preprocess_query(query):
    tokens = nltk.word_tokenize(query.lower())
    return [stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words]

# Normalizes TF-IDF scores between 0 and 1 for fair ranking.
def normalize_scores(results):
    if not results:
        return results
    max_score = max(results.values())
    return {doc: score / max_score for doc, score in results.items()}

# Boosts ranking for faculty and research pages.
def faculty_boost(doc, query_terms):
    boost = 1.0

    # Higher weight for ANY university-related pages (not just UCI)
    if ".edu" in doc:
        boost += 1.5  # Boost for any university site

    # Give additional boost if the query terms (name) appear in the URL
    for term in query_terms:
        if term in doc.lower():
            boost += 2.0  # Strong boost if the name is in the URL

    # Boost personal pages, for ex ~username/ format used for faculty pages)
    if "/~" in doc:
        boost += 2.5  # Stronger boost for faculty profile pages

    return boost

# Boosts ranking if query terms appear close together in the document.
def proximity_boost(term_positions):
    if not term_positions or len(term_positions) < 2:
        return 1  # No boost if there's only one term

    min_distance = float('inf')
    sorted_positions = sorted(term_positions)

    for i in range(1, len(sorted_positions)):
        min_distance = min(min_distance, sorted_positions[i] - sorted_positions[i - 1])

    return 1 + (1 / (1 + min_distance))  # Smaller distance = higher boost

def spell_correct(query_terms, index_terms):
    corrected_terms = []
    corrections_made = False

    # Attempting to correct the entire phrase while maintaining spaces
    full_query = " ".join(query_terms)
    suggested_phrase = get_close_matches(full_query, [" ".join(index_terms)], n=1, cutoff=0.85)

    if suggested_phrase:
        return suggested_phrase[0].split(), True  # Keep words separate

    # If no phrase match, correct words individually
    for term in query_terms:
        if term in index_terms:
            corrected_terms.append(term)
        else:
            suggestion = get_close_matches(term, index_terms, n=1, cutoff=0.8)
            if suggestion:
                corrected_terms.append(suggestion[0])  # Keep words separate
                corrections_made = True
            else:
                corrected_terms.append(term)

    return corrected_terms, corrections_made

def search(query):
    if not inverted_index:
        print("\n The index is empty. Please ensure the index file is correctly generated.")
        return

    start_time = time.time()

    query_terms = preprocess_query(query)

    # Applying spell correction
    corrected_terms, corrections_made = spell_correct(query_terms, inverted_index.keys())

    if corrections_made:
        corrected_query = " ".join(corrected_terms)
        print(f"\n Did you mean: {corrected_query}? Searching for corrected term...\n")
        query_terms = corrected_terms  # Use corrected query

    results = defaultdict(float)
    query_vector = defaultdict(float)

    # Computing query term frequencies
    for term in query_terms:
        query_vector[term] += 1

    # Normaliz query TF
    for term in query_vector:
        query_vector[term] = 1 + math.log(query_vector[term])

    # Retrieving relevant documents
    for term in query_terms:
        if term in inverted_index:
            doc_count = len(inverted_index[term])
            idf = math.log((len(inverted_index) + 1) / (1 + doc_count))

            for doc, info in inverted_index[term].items():
                # Handle both cases: info as a dictionary or a float (direct TF-IDF value)
                if isinstance(info, dict):  
                    tfidf_score = query_vector[term] * info["tf"] * idf
                    proximity_bonus = proximity_boost(info.get("positions", []))  # Apply proximity boost
                else:  
                    tfidf_score = query_vector[term] * info * idf 
                    proximity_bonus = 1  # No proximity bonus since positions are unavailable
                
                # Boosting faculty and research pages
                boost = faculty_boost(doc, query_terms)
                results[doc] += tfidf_score * boost * proximity_bonus

    # Normalizing scores
    results = normalize_scores(results)

    # Ranking results
    ranked_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000

    # Displaying results
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
