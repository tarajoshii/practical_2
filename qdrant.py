import ollama
import numpy as np
import time
import psutil
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

process = psutil.Process(os.getpid())


# Initialize Qdrant connection
qdrant_client = QdrantClient("localhost", port=6333)

VECTOR_DIM = 768
COLLECTION_NAME = "embedding_collection"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# Create collection if not exists
qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
)

# Generate an embedding using Ollama
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# Store the calculated embedding in Qdrant
def store_embedding(doc_id: str, text: str, embedding: list):
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=int(doc_id),
                vector=embedding,
                payload={"text": text}
            )
        ]
    )
    print(f"Stored embedding for: {text}")

# Generate a response using Ollama and retrieved Qdrant context
def generate_response(query_text: str, model: str = "mistral"):
    # Generate an embedding for the query
    embedding = get_embedding(query_text)

    # Perform vector search in Qdrant
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=3
    )

    # Extract top matching documents
    context_texts = [hit.payload["text"] for hit in search_results]
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
