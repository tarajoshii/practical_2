import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data
filename = "my_experiments.csv"
df = pd.read_csv(filename, on_bad_lines='skip', engine='python')

# Function to create grouped bar chart for a given DB
def plot_db_comparison(df, db_name):
    # Filter for this DB
    db_df = df[df["db"] == db_name].copy()
    db_df["Config"] = db_df["embedding_model"] + " + " + db_df["llm_model"]
    
    # Use absolute memory values
    db_df["avg_memory_MB"] = (
        abs(db_df["embedding_memory_MB"]) + abs(db_df["query_memory_MB"])
    ) / 2

    # Metrics
    categories = ["embedding_time_sec", "query_time_sec", "avg_memory_MB"]
    labels = ["Avg Embedding Time (s)", "Avg Query Time (s)", "Avg Memory Usage (MB)"]
    x = np.arange(len(db_df))
    bar_width = 0.25

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, category in enumerate(categories):
        ax.bar(x + i * bar_width, db_df[category], width=bar_width, label=labels[i])

    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(db_df["Config"], rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(f"{db_name.capitalize()} Pipeline Comparison – Time & Memory")
    ax.legend()
    plt.tight_layout()
    plt.show()

# Plot for each DB
for db in ["redis", "chroma", "qdrant"]:
    plot_db_comparison(df, db)
