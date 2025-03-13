import ollama
import redis
import numpy as np
from redis.commands.search.query import Query

# Initialize Redis connection
redis_client = redis.Redis(host="localhost", port=6380, db=0)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


# Create an index in Redis
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")


# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# Store the calculated embedding in Redis
def store_embedding(doc_id: str, text: str, embedding: list):
    key = f"{DOC_PREFIX}{doc_id}"
    redis_client.hset(
        key,
        mapping={
            "text": text,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  # Store as byte array
        },
    )
    print(f"Stored embedding for: {text}")


# Generate a response using Ollama and retrieved Redis context
def generate_response(query_text: str, model: str = "mistral"):
    # Generate an embedding for the query
    embedding = get_embedding(query_text)

    # Perform vector search in Redis
    q = (
        Query("*=>[KNN 3 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("text", "vector_distance")
        .dialect(2)
    )

    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )

    # Extract top matching documents
    context_texts = [doc.text for doc in res.docs]
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
    create_hnsw_index()

    # Example texts to encode and store
    texts = [
        "Redis is an in-memory key-value database.",
        "Ollama provides efficient LLM inference on local machines.",
        "Vector databases store high-dimensional embeddings for similarity search.",
        "HNSW indexing enables fast vector search in Redis.",
        "Ollama can generate embeddings for RAG applications.",
    ]

    for i, text in enumerate(texts):
        embedding = get_embedding(text)
        store_embedding(str(i), text, embedding)

    # Example query and AI-generated response
    query = "How does Redis perform vector searches?"
    answer = generate_response(query)
    print("\n🔹 Ollama Generated Response:\n", answer)
