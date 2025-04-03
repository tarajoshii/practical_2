# DS4300: Practical 2-Vector DBs & LLMs
## Created By: Aadya, Tara, and Shruti

## Project Overview

This project evaluates different embedding models and vector databases to determine the best pipeline for efficiently indexing and querying lecture notes. The primary objectives are to measure:
  - **Indexing Time:** How long it takes to store embeddings in each vector database.
  - **Query Time:** How quickly each database retrieves relevant information.
  - **Memory Usage:** The impact of different embedding models and databases on system resources.
  - **Overall Efficiency:** Identifying the optimal combination for specific use cases.

We compare performance across Redis, ChromaDB, and Qdrant when paired with embedding models such as nomic-embed-text, mxbai-embed-large, and all-minilm.

## Installation & Dependencies

### Prerequisites
- Python 3.7 or higher
- Following Services:
  - Redis
  - Qdrant
  - ChromaDB

### Required Libraries
Install The Following Libraries:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - Any other libraries needed for your vector database integrations

## How to Run
1. **Preprocess Lecture Notes**  
   Run `preproc.py` to clean and standardize your lecture notes data. This step ensures that all data is in the correct format for the embedding process.
   ```bash
   python preproc.py
   ```
2. **Generate Embeddings**  
   Execute `mxbai_embed_large_sample.py` to process the preprocessed data through the selected embedding models. You can choose between model by modifying the global variable at the top of the script.
   ```bash
   python mxbai_embed_large_sample.py
   ```
3. **Index Data in Vector Databases**  
   Index the generated embeddings using your preferred vector database by running one of the following scripts:
   - **Redis:**  
     ```bash
     python redis_sample.py
     ```
   - **ChromaDB:**  
     ```bash
     python chroma.py
     ```
   - **Qdrant:**  
     ```bash
     python qdrant.py
     ```
4. **Querying Capabilities**  
   Use `argeparser_queries.py` to interact with the vector databases via command-line queries. This script allows you to test and compare query performance across the different databases.
   ```bash
   python argeparser_queries.py
   ```
5. **Experimental Analysis & Visualization**  
   - Run `experiments.py` to perform performance evaluations and measure key metrics such as indexing time and query response time.
   - Execute `experimental_analysis.py` to provide a detailed analysis of the experimental results.
   - Finally, use `pipeline_viz.py` to generate visualizations of the data processing pipeline and overall performance metrics.
     
   Additionally, you can run all experiments across all configurations with the following command:
   ```bash
   python experiments.py --db all --embedding_model all --llm_model all --outfile my_experiments.csv
   ```
   This command evaluates every combination of vector database, embedding model, and LLM model, outputting the results to `my_experiments.csv`.

### Example Execution
To evaluate the performance using ChromaDB with the MiniLM embedding model and using Mistral as the LLM:
1. Set the Embedding Model:
   - In mxbai_embed_large_sample.py, set the global variable for the embedding model to MiniLM.
2. Set the LLM Model:
   - In the experimental configuration (or in the query generation function within the scripts), set the global variable for         the LLM to Mistral (or change it to Llama 2 7b as needed).
3. Index Embeddings:
   - Run chroma.py to index the generated embeddings in ChromaDB.
4. Query Data:
   - Execute argeparser_queries.py to run queries against ChromaDB.
5. Analyze & Visualize:
   - Use experiments.py and pipeline_viz.py to analyze and visualize the indexing and query performance.

