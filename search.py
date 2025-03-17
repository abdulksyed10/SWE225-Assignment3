import json
import time
import nltk
import os
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import defaultdict
import math
from difflib import get_close_matches
from spellchecker import SpellChecker

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
        boost += 1  # Boost for any university site
    if ".html" in doc:
        boost -= 0.5

    # Give additional boost if the query terms (name) appear in the URL
    for term in query_terms:
        if term in doc.lower():
            boost += 2.0  # Strong boost if the name is in the URL

    if "/~" in doc:
        url_parts = doc.split("/~")
        if len(url_parts) > 1:  # Ensure there's a username after "/~"
            username_part = url_parts[1].split("/")[0].lower()
            if any(term.lower() == username_part for term in query_terms):
                boost *= 500  # Apply extra boost ONLY if the username matches a query term

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


# Spell Correction using pyspellchecker
spell = SpellChecker()
def correct_spelling(query):
    words = query.split()
    corrected_words = []

    for word in words:
        # If the word is in the index, it's likely a valid name or term, so don't correct it
        if word.lower() in inverted_index:
            corrected_words.append(word)
        else:
            correction = spell.correction(word)
            corrected_words.append(correction if correction else word)  # Keep original if None
    
    corrected_query = " ".join(corrected_words)
    return corrected_query if corrected_query != query else None

def search(query):
    if not inverted_index:
        return [], 0, None  # Ensure function returns proper values if index is empty

    start_time = time.time()

    # Apply spell correction
    query_terms = preprocess_query(query)

    results = defaultdict(float)
    query_vector = defaultdict(float)

    # Compute query term frequencies
    for term in query_terms:
        query_vector[term] += 1

    # Normalize query TF
    for term in query_vector:
        query_vector[term] = 1 + math.log(query_vector[term])

    # Retrieve relevant documents
    for term in query_terms:
        if term in inverted_index:
            doc_count = len(inverted_index[term])
            idf = math.log((len(inverted_index) + 1) / (1 + doc_count))

            for doc, info in inverted_index[term].items():
                if isinstance(info, dict):  
                    tfidf_score = query_vector[term] * info["tf"] * idf
                    proximity_bonus = proximity_boost(info.get("positions", []))
                else:  
                    tfidf_score = query_vector[term] * info * idf  
                    proximity_bonus = 1  

                boost = faculty_boost(doc, query_terms)
                results[doc] += tfidf_score * boost * proximity_bonus

    # Normalize and rank results
    results = normalize_scores(results)
    ranked_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # Convert to ms

    return ranked_results, elapsed_time
