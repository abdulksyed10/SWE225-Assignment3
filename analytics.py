import json

# Define the index file
MERGED_INDEX_FILE = "final_inverted_index.json"

def count_index_stats():
    """Counts the number of documents and unique tokens in the final inverted index."""
    try:
        with open(MERGED_INDEX_FILE, "r", encoding="utf-8") as f:
            inverted_index = json.load(f)
        
        num_tokens = len(inverted_index)  # Unique tokens (keys in dictionary)
        num_documents = set()  # Unique document URLs

        # Count unique documents across all tokens
        for postings in inverted_index.values():
            num_documents.update(postings.keys())  # Add document URLs to set

        print(f"Total Unique Tokens: {num_tokens}")
        print(f"Total Unique Documents: {len(num_documents)}")

    except FileNotFoundError:
        print(f"Error: {MERGED_INDEX_FILE} not found. Run the indexer first.")
    except json.JSONDecodeError:
        print(f"Error: Failed to parse {MERGED_INDEX_FILE}. Ensure it's a valid JSON file.")

if __name__ == "__main__":
    count_index_stats()
