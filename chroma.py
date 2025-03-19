import ollama
import numpy as np
import time
import psutil
import os
import chromadb

process = psutil.Process(os.getpid())

# Initialize Chroma client
# chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_client = chromadb.Client()

COLLECTION_NAME = "embedding_collection"

# Create or get collection
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

VECTOR_DIM = 768

# Generate an embedding using Ollama
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# Store the calculated embedding in Chroma
def store_embedding(doc_id: str, text: str, embedding: list):
    collection.add(
        documents=[text],
        metadatas=[{"doc_id": doc_id}],
        embeddings=[embedding],
        ids=[doc_id]
    )
    print(f"Stored embedding for: {text}")

# Generate a response using Ollama and retrieved Chroma context
def generate_response(query_text: str, model: str = "mistral"):
    # Generate an embedding for the query
    embedding = get_embedding(query_text)

    # Perform vector search in Chroma
    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )

    # Extract top matching documents
    context_texts = results['documents'][0] if results['documents'] else []
    context = "\n\n".join(context_texts)

    # Construct the prompt for Ollama
    prompt = f"""
    You are an AI assistant. Use the following retrieved context to answer the question accurately:

    Context:
    {context}

    Question: {query_text}

    Answer:
    """

    # Generate a response using Ollama
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

if __name__ == "__main__":
    start_time = time.time()
    memory_before = process.memory_info().rss / 1024 ** 2  # Convert to MB

    # Example texts to encode and store
    texts = [
        "Redis is an in-memory key-value database.",
        "Ollama provides efficient LLM inference on local machines.",
        "Vector databases store high-dimensional embeddings for similarity search.",
        "HNSW indexing enables fast vector search in Redis.",
        "Ollama can generate embeddings for RAG applications.",
    ]

    embed_start_time = time.time()
    embed_memory_before = process.memory_info().rss / 1024 ** 2  
    for i, text in enumerate(texts):
        embedding = get_embedding(text)
        store_embedding(str(i), text, embedding)

    embed_end_time = time.time()
    embed_memory_after = process.memory_info().rss / 1024 ** 2 

    print(f"\nEmbedding Execution Time: {embed_end_time - embed_start_time:.4f} seconds")
    print(f"Embedding Memory Usage: {embed_memory_after - embed_memory_before:.2f} MB")

    # Example query and AI-generated response
    query = "How does Redis perform vector searches?"

    query_start_time = time.time()
    query_memory_before = process.memory_info().rss / 1024 ** 2  
    answer = generate_response(query)
    print("\n🔹 Ollama Generated Response:\n", answer)
    
    query_end_time = time.time()
    query_memory_after = process.memory_info().rss / 1024 ** 2 

    print(f"\nQuery Execution Time: {query_end_time - query_start_time:.4f} seconds")
    print(f"Query Memory Usage: {query_memory_after - query_memory_before:.2f} MB")

    # End of script
    end_time = time.time()
    memory_after = process.memory_info().rss / 1024 ** 2  # Convert to MB
    print(f"\nExecution Time: {end_time - start_time:.4f} seconds")
    print(f"Memory Usage: {memory_after - memory_before:.2f} MB")
