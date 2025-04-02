import pandas as pd

# Path to CSV file
filename = "my_experiments.csv"

# Read the CSV file (skip any rows with extra fields)
df = pd.read_csv(filename, on_bad_lines='skip', engine='python')

# Calculate averages by embedding model
embedding_averages = df.groupby("embedding_model").agg({
    "embedding_time_sec": "mean",
    "query_time_sec": "mean",
    "total_memory_MB": "mean"
}).reset_index()

print("Averages by Embedding Model:")
print(embedding_averages)

# Calculate averages by LLM model
llm_averages = df.groupby("llm_model").agg({
    "embedding_time_sec": "mean",
    "query_time_sec": "mean",
    "total_memory_MB": "mean"
}).reset_index()

print("\nAverages by LLM Model:")
print(llm_averages)

# Calculate averages by Vector Database (db)
db_averages = df.groupby("db").agg({
    "embedding_time_sec": "mean",
    "query_time_sec": "mean",
    "total_memory_MB": "mean"
}).reset_index()

print("\nAverages by Vector Database:")
print(db_averages)
