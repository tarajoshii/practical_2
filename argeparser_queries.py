import argparse
import ollama
import redis
import chromadb
import numpy as np
import time
import psutil
import os
from redis.commands.search.query import Query
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import redis
import chromadb
from qdrant_client import QdrantClient
import numpy as np
import redis
import chromadb
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance


# Constants

VECTOR_DIM = 768
DOC_PREFIX = "doc:"
COLLECTION_NAME = "embedding_collection"
DISTANCE_METRIC = "COSINE"
INDEX_NAME = "embedding_index"

# Client Initialization Functions
def initialize_redis():
    return redis.Redis(host="localhost", port=6380, db=0)

def initialize_chroma():
    return chromadb.Client()

def initialize_qdrant():
    return QdrantClient("localhost", port=6333)

# Mapping Client Initialization
CLIENTS = {
    'redis': initialize_redis,
    'chroma': initialize_chroma,
    'qdrant': initialize_qdrant
}

def initialize_clients(db):
    client_initializer = CLIENTS.get(db)
    if client_initializer:
        return client_initializer()
    else:
        print(f"Client for '{db}' does not exist.")
        return None

# Common Store Embedding Function
def store_embedding(client, db, doc_id, text, embedding):
    if db == 'redis':
        key = f"{DOC_PREFIX}{doc_id}"
        client.hset(
            key,
            mapping={
                "text": text,
                "embedding": np.array(embedding, dtype=np.float32).tobytes()
            }
        )
    elif db == 'chroma':
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        collection.add(
            documents=[text],
            metadatas=[{"doc_id": doc_id}],
            embeddings=[embedding],
            ids=[doc_id]
        )
    elif db == 'qdrant':
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(id=int(doc_id), vector=embedding, payload={"text": text})]
        )
    else:
        print(f"Database {db} not supported.")
        return
    print(f"Stored embedding for: {text}")

# Common Get Context Function
def get_context(client, db, embedding):
    if db == 'redis':
        query = (
            Query("*=>[KNN 3 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("text", "vector_distance")
            .dialect(2)
        )
        res = client.ft(INDEX_NAME).search(
            query, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
        )
        context_texts = [doc.text for doc in res.docs]
    elif db == 'chroma':
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=3
        )
        context_texts = results['documents'][0] if results['documents'] else []
    elif db == 'qdrant':
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=embedding,
            limit=3
        )
        context_texts = [hit.payload["text"] for hit in search_results]
    else:
        print(f"Database {db} not supported.")
        return ""
    
    return "\n\n".join(context_texts)

# Function to Generate Response
def generate_response(db, client, embedding, query_text: str, llm_model: str):
    context = get_context(client, db, embedding)
    
    # Construct prompt
    prompt = f"""
    You are an AI assistant. Use the following retrieved context to answer the question accurately:

    Context:
    {context}

    Question: {query_text}

    Answer:
    """
    response = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Function to get embedding (unchanged)
def get_embedding(text: str, embedding_model: str) -> list: #nomic-embed-text
    response = ollama.embeddings(model=embedding_model, prompt=text)
    return response["embedding"]


def get_answers(db, embedding_model, llm_model):
    client = initialize_clients(db)
    if not client:
        print("Failed to initialize client.")
        return
    
    process = psutil.Process(os.getpid())
    start_time = time.time()
    memory_before = process.memory_info().rss / 1024 ** 2  # Convert to MB
    
    # Example texts to store and generate embeddings
    texts = [
        "Redis is an in-memory key-value database.",
        "Ollama provides efficient LLM inference on local machines.",
        "Vector databases store high-dimensional embeddings for similarity search.",
        "HNSW indexing enables fast vector search in Redis.",
        "Ollama can generate embeddings for RAG applications.",
    ]
    
    embed_start_time = time.time()
    embed_memory_before = process.memory_info().rss / 1024 ** 2  # Memory usage before embedding
    for i, text in enumerate(texts):
        embedding = get_embedding(text, embedding_model)
        store_embedding(client, db, str(i), text, embedding)  # Store embedding in the appropriate DB
    embed_end_time = time.time()
    embed_memory_after = process.memory_info().rss / 1024 ** 2  # Memory usage after embedding
    
    print(f"\nEmbedding Execution Time: {embed_end_time - embed_start_time:.4f} seconds")
    print(f"Embedding Memory Usage: {embed_memory_after - embed_memory_before:.2f} MB")

    # Example query and AI-generated response
    query = "How does Redis perform vector searches?"
    query_start_time = time.time()
    query_memory_before = process.memory_info().rss / 1024 ** 2  # Memory usage before query
    answer = generate_response(db, client, embedding, query, llm_model)  # Pass db and client
    print("\n🔹 Ollama Generated Response:\n", answer)
    
    query_end_time = time.time()
    query_memory_after = process.memory_info().rss / 1024 ** 2  # Memory usage after query
    
    print(f"\nQuery Execution Time: {query_end_time - query_start_time:.4f} seconds")
    print(f"Query Memory Usage: {query_memory_after - query_memory_before:.2f} MB")

    # Final execution time and memory usage
    end_time = time.time()
    memory_after = process.memory_info().rss / 1024 ** 2  # Final memory usage
    print(f"\nTotal Execution Time: {end_time - start_time:.4f} seconds")
    print(f"Total Memory Usage: {memory_after - memory_before:.2f} MB")


# Argument parser setup
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the vector search and LLM model.")
    parser.add_argument('--db', type=str, choices=['redis', 'chroma', 'qdrant'], required=True, help="Database to use")
    parser.add_argument('--embedding_model', type=str, choices=['mxbai-embed-large', 'nomic-embed-text'], required=True, help="Embedding model to use")
    parser.add_argument('--llm_model', type=str, choices=['mistral','llama-27b'], required=True, help="LLM model to use")
    
    args = parser.parse_args()
    
    get_answers(args.db, args.embedding_model, args.llm_model)
    
    # ex: python shared.py --db chroma --embedding_model nomic-embed-text --llm_model mistral  