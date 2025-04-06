# PubMed Retrieval System & Evaluation Pipeline

This project implements a pipeline for fetching biomedical abstracts from PubMed, generating relevant queries, creating relevance judgments (using TF-IDF and optionally Large Language Models), evaluating various information retrieval models, and analyzing the results.

The pipeline supports:
- Fetching data from PubMed based on topics.
- Generating additional queries using templates or LLM prompts.
- Creating relevance judgments using TF-IDF or LLM calls (Google AI / Vertex AI).
- Analyzing and refining the generated query set.
- Evaluating retrieval models: BM25, various Dense Retrievers (Sentence Transformers), and Hybrid combinations.
- Saving evaluation results.

## Setup

1.  **Clone the Repository:**
    ```bash
    # git clone <your-repo-url>
    # cd <your-repo-directory>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # Activate the environment
    # Windows:
    .\.venv\Scripts\activate
    # macOS/Linux:
    # source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Access:**

    *   **PubMed:** No API key is strictly required for basic use, but providing your email in `config.yaml` (`pubmed_api.email`) is recommended by NCBI guidelines.
    *   **LLM for Relevance (Optional):** If using the LLM for relevance judgments (`llm.enabled_relevance: true` in `config.yaml`), configure either Google AI or Vertex AI:
        *   **Google AI (Gemini API Key):**
            1.  Create a `.env` file in the project root.
            2.  Add your Google AI API key to the `.env` file:
                ```
                GOOGLE_API_KEY="YOUR_API_KEY_HERE"
                ```
            3.  Ensure `llm.api_provider` is set to `google` in `config.yaml`.
        *   **Vertex AI (Google Cloud):**
            1.  Ensure you have the Google Cloud SDK installed and configured.
            2.  Authenticate using Application Default Credentials (ADC):
                ```bash
                gcloud auth application-default login
                ```
            3.  Ensure your Google Cloud project has the Vertex AI API enabled and sufficient quotas.
            4.  Set `llm.api_provider` to `vertex_ai` in `config.yaml`.
            5.  Update `vertex_ai_project` and `vertex_ai_location` in `config.yaml` with your Google Cloud project ID and desired region (e.g., `us-central1`).

## Configuration (`config.yaml`)

This file controls the entire pipeline. Key sections include:

*   `pubmed_api`: Settings for interacting with NCBI PubMed (email, batch sizes, delays).
*   `dataset_generation`: Parameters for initial dataset creation (topics, max documents).
*   `query_generation`: Settings for generating additional queries (templates, topic extraction/validation, relevance update method).
*   `query_analysis`: Parameters for cleaning/filtering the generated queries.
*   `retrieval`: Configuration for retrieval models (paths to models, hybrid alpha values).
*   `llm`: Settings for using Large Language Models (provider choice, API keys/project info, model names, prompts, rate limiting delay).
*   `paths`: Input/output file paths for datasets and results.
*   `evaluation`: Metrics and parameters (e.g., k values) for evaluation.

**Important:** Review and modify `config.yaml` carefully before running the pipeline, especially API settings and file paths.

## Project Structure

*   `fetch_pubmed.py`: Fetches abstracts from PubMed, generates initial queries, and optionally creates LLM relevance judgments for the base set.
*   `generate_queries.py`: Extracts topics, generates additional queries, and updates relevance judgments (TF-IDF or LLM) for the expanded set.
*   `analyze_and_improve_queries.py`: Analyzes the query set (e.g., deduplication) and produces the final dataset.
*   `evaluate_retrieval.py`: Evaluates configured retrieval models on the final dataset.
*   `retrieval_models.py`: Contains implementations of BM25, Dense, and Hybrid retrieval models.
*   `config.yaml`: Main configuration file for the pipeline.
*   `requirements.txt`: Python dependencies.
*   `run_pipeline.bat`: Batch script to execute the full pipeline sequentially.
*   `.env` (if created): Stores the Google AI API key.
*   `*.json`: Dataset files (e.g., `pubmed_dataset.json`, `pubmed_dataset_expanded.json`, `pubmed_dataset_final.json`, `evaluation_results.json`).
*   `README.md`: This file.

## Pipeline Execution

You can run the pipeline steps individually or execute the entire sequence.

1.  **Run Full Pipeline:**
    ```bash
    .\run_pipeline.bat
    ```
    This script runs `fetch_pubmed.py`, `generate_queries.py`, and `analyze_and_improve_queries.py` in order, using the settings in `config.yaml`. It reports warnings on errors but attempts to continue.

2.  **Run Individual Steps:**
    ```bash
    # Step 1: Fetch initial data and relevance
    python fetch_pubmed.py --config config.yaml

    # Step 2: Generate more queries and update relevance
    python generate_queries.py --config config.yaml

    # Step 3: Analyze and finalize dataset
    python analyze_and_improve_queries.py --config config.yaml

    # Step 4: Evaluate retrieval models
    python evaluate_retrieval.py --config config.yaml
    ```

## Evaluation

The `evaluate_retrieval.py` script loads the dataset specified by `paths.final_dataset_path` in `config.yaml`. It initializes and evaluates the retriever models defined in the `retrieval` section of the config.

Key metrics calculated include:
- Precision@k
- Recall@k
- NDCG@k (k=1, 3, 5, 10, 20)
- MAP (Mean Average Precision)
- MRR (Mean Reciprocal Rank)

Results are printed to the console and saved to the file specified by `paths.results_file` (e.g., `evaluation_results.json`).

## Models

1. **BM25 Retriever**
   - Traditional lexical retrieval model
   - Good for exact keyword matching

2. **Dense Retriever**
   - Semantic retrieval using sentence transformers
   - Better for understanding query intent and semantic similarity

3. **Hybrid Retriever**
   - Combines BM25 and Dense retrieval
   - Adjustable mixing ratio (alpha parameter)
   - Can leverage both lexical and semantic matching

## Evaluation Metrics

The system evaluates retrievers using:
- Precision: Accuracy of retrieved documents
- Recall: Coverage of relevant documents
- NDCG: Ranking quality considering document positions

## Customization

You can:
1. Modify the medical dataset in `medical_dataset.json`
2. Adjust the hybrid mixing ratio (alpha) in the HybridRetriever
3. Change the number of retrieved documents (k) in the evaluation
4. Add new retrieval models by extending the BaseRetriever class 