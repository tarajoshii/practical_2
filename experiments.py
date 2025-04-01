import argparse
import ollama
import redis
import chromadb
import numpy as np
import time
import psutil
import os
import pandas as pd
from redis.commands.search.query import Query
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from itertools import product
from sentence_transformers import SentenceTransformer


# Constants
VECTOR_DIM = 768
DOC_PREFIX = "doc:"
COLLECTION_NAME = "embedding_collection"
DISTANCE_METRIC = "COSINE"
INDEX_NAME = "embedding_index"

file_path = 'my_experiments.csv'

# Step 1: Load existing data once
if os.path.exists(file_path):
    existing_data = pd.read_csv(file_path)
else:
    existing_data = pd.DataFrame()

# Client Initialization Functions
def initialize_redis():
    return redis.Redis(host="localhost", port=6379, db=0)

def initialize_chroma():
    return chromadb.Client()

def initialize_qdrant():
    client = QdrantClient(host="localhost", port=6333)

    # does collection exist
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' does not exist. Creating collection.")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

    return client

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

def generate_response(db, client, embedding, query_text: str, llm_model: str):
    context = get_context(client, db, embedding)
    prompt = f"""
    You are an AI assistant. Use the following retrieved context to answer the question accurately:

    Context:
    {context}

    Question: {query_text}

    Answer:
    """
    response = ollama.chat(model=llm_model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def get_embedding(text: str, embedding_model: str) -> list:
    response = ollama.embeddings(model=embedding_model, prompt=text)
    return response["embedding"]

def run_experiment(db, embedding_model, llm_model):
    client = initialize_clients(db)
    if not client:
        print("Failed to initialize client.")
        return None

    process = psutil.Process(os.getpid())
    start_time = time.time()
    memory_before = process.memory_info().rss / 1024 ** 2

    texts = [
        "Redis is an in-memory key-value database.",
        "Ollama provides efficient LLM inference on local machines.",
        "Vector databases store high-dimensional embeddings for similarity search.",
        "HNSW indexing enables fast vector search in Redis.",
        "Ollama can generate embeddings for RAG applications."
    ]

    embed_start_time = time.time()
    embed_memory_before = process.memory_info().rss / 1024 ** 2
    for i, text in enumerate(texts):
        embedding = get_embedding(text, embedding_model)
        store_embedding(client, db, str(i), text, embedding)
    embed_end_time = time.time()
    embed_memory_after = process.memory_info().rss / 1024 ** 2

    query = "How does Redis perform vector searches?"
    query_start_time = time.time()
    query_memory_before = process.memory_info().rss / 1024 ** 2
    answer = generate_response(db, client, embedding, query, llm_model)
    query_end_time = time.time()
    query_memory_after = process.memory_info().rss / 1024 ** 2

    end_time = time.time()
    memory_after = process.memory_info().rss / 1024 ** 2

    return {
        "db": db,
        "embedding_model": embedding_model,
        "llm_model": llm_model,
        "embedding_time_sec": embed_end_time - embed_start_time,
        "embedding_memory_MB": embed_memory_after - embed_memory_before,
        "query_time_sec": query_end_time - query_start_time,
        "query_memory_MB": query_memory_after - query_memory_before,
        "total_time_sec": end_time - start_time,
        "total_memory_MB": memory_after - memory_before,
        "answer_snippet": answer[:100]
    }

def append_unique_rows(file_path, new_data):
    # Check if the file exists
    if os.path.exists(file_path):
        # Read existing data
        existing_data = pd.read_csv(file_path)
        # Concatenate existing data with new data and drop duplicates
        combined_data = pd.concat([existing_data, new_data]).drop_duplicates()
    else:
        # If file doesn't exist, new data becomes the combined data
        combined_data = new_data
    
    # Write the combined data back to the CSV file
    combined_data.to_csv(file_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vector + LLM experiments")
    parser.add_argument('--db', type=str, help="Database to use (or 'all')", default="all")
    parser.add_argument('--embedding_model', type=str, help="Embedding model to use (or 'all')", default="all")
    parser.add_argument('--llm_model', type=str, help="LLM model to use (or 'all')", default="all")
    parser.add_argument('--outfile', type=str, default="experiment_results.csv")
    args = parser.parse_args()

    dbs = ["redis", "chroma", "qdrant"] if args.db == "all" else [args.db]
    embedding_models = ["mxbai-embed-large", "nomic-embed-text", "all-minilm"] if args.embedding_model == "all" else [args.embedding_model]
    llm_models = ["mistral", "llama2"] if args.llm_model == "all" else [args.llm_model]

    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as file:

        for db, emb_model, llm_model in product(dbs, embedding_models, llm_models):
            print(f"\n▶️ Running with db={db}, embedding_model={emb_model}, llm_model={llm_model}")
            result = run_experiment(db, emb_model, llm_model)

            if result:
                new_data = pd.DataFrame([result])
                new_data.to_csv(file, index=False, header=not file_exists)
                file_exists = True

    # # Convert results to DataFrame
    # new_data = pd.DataFrame(results)
    
    # # Append unique rows to the CSV file
    # append_unique_rows(args.outfile, new_data)
    # df = pd.DataFrame(results)
    # df.to_csv(args.outfile, index=False)
    print(f"\n✅ Results saved to {args.outfile}")
