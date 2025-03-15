def normalize_line_endings(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()

    # Replace CRLF (carriage return + line feed) with LF only
    normalized_data = data.replace("\r\n", "\n")

    # Rewrite the file with normalized line endings
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(normalized_data)
    print(f"Normalized line endings in {file_path}")

# Apply normalization to the partial index file
normalize_line_endings("partial_indexes/partial_index_277.json")
