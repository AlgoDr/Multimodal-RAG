def chunk_text_by_words(text, chunk_size, overlap):
    """
    Splits a large text into smaller chunks based on word count,
    with a specified overlap to preserve context between chunks.
    
    Args:
        text (str): The large text to be chunked.
        chunk_size (int): The number of words in each chunk.
        overlap (int): The number of words to overlap between consecutive chunks.
        
    Returns:
        list of str: The list of text chunks.
    """
    # Split the text into a list of words
    words = text.split()
    chunks = []
    
    # Ensure overlap is not larger than or equal to chunk_size to avoid infinite loops
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk_size")
    
    # Step size is how many words we move forward for the next chunk
    step_size = chunk_size - overlap
    
    # Iterate through the words using the step size
    for i in range(0, len(words), step_size):
        # Extract a slice of words for the current chunk
        chunk_words = words[i : i + chunk_size]
        
        # Join the words back into a single string
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        
        # If the chunk we just took reaches or goes past the end of the text, stop
        if i + chunk_size >= len(words):
            break
            
    return chunks


Corpus_data = "loreum ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."


# --- Example Usage ---
if __name__ == "__main__":
    
    
    CHUNK_SIZE = 15
    OVERLAP = 5
    
    print(f"Original Text Word Count: {len(Corpus_data.split())}\n")
    
    chunks = chunk_text_by_words(Corpus_data, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    
    for idx, chunk in enumerate(chunks):
        print(f"--- Chunk {idx + 1} (Words: {len(chunk.split())}) ---")
        print(chunk)
        print()
