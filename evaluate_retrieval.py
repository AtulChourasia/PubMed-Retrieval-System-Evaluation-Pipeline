import json
from typing import List, Dict, Tuple, Set
import numpy as np
from tqdm import tqdm
import argparse
import yaml
import logging
import torch

# Import retrievers and the dataset loader
from retrieval_models import (BM25Retriever, DenseRetriever, HybridRetriever,
                              MedicalDenseRetriever, ClinicalDenseRetriever,
                              BioBERTDenseRetriever, PubMedBERTDenseRetriever,
                              BaseRetriever, CrossEncoderReRanker)
# Assuming load_dataset handles the dict format and validation
from generate_queries import load_dataset

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Evaluation Metrics ---
# (Functions calculate_precision_at_k, calculate_recall_at_k, calculate_ndcg_at_k, calculate_map, calculate_mrr remain largely the same,
# but we add type hints and minor checks)

def calculate_precision_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int) -> float:
    """Calculate precision@k."""
    if k == 0: return 0.0
    retrieved_k = retrieved_doc_ids[:k]
    if not retrieved_k: return 0.0 # Handle empty retrieval
    relevant_retrieved_count = len(set(retrieved_k) & relevant_doc_ids)
    return relevant_retrieved_count / len(retrieved_k) # Use actual number retrieved up to k

def calculate_recall_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int) -> float:
    """Calculate recall@k."""
    if not relevant_doc_ids: return 1.0 # Or 0.0? Convention varies. Assume 1.0 if nothing is relevant.
    if k == 0: return 0.0
    retrieved_k = retrieved_doc_ids[:k]
    relevant_retrieved_count = len(set(retrieved_k) & relevant_doc_ids)
    return relevant_retrieved_count / len(relevant_doc_ids)

def calculate_ndcg_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str], k: int) -> float:
    """Calculate NDCG@k."""
    if k == 0: return 0.0
    dcg = 0.0
    # Calculate DCG for retrieved list
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        # Assume binary relevance (1 if relevant, 0 otherwise)
        relevance = 1 if doc_id in relevant_doc_ids else 0
        dcg += relevance / np.log2(i + 2) # Discount factor starts at log2(2)

    # Calculate IDCG (Ideal DCG)
    idcg = 0.0
    num_relevant = len(relevant_doc_ids)
    for i in range(min(num_relevant, k)):
        idcg += 1 / np.log2(i + 2) # Ideal ranking puts all relevant items first

    return dcg / idcg if idcg > 0 else 0.0

def calculate_map(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str]) -> float:
    """Calculate Mean Average Precision (MAP) for a single query."""
    if not relevant_doc_ids: return 1.0 # Or 0.0? Consistent with recall: 1.0 if nothing is relevant.

    ap = 0.0
    num_relevant_found = 0
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in relevant_doc_ids:
            num_relevant_found += 1
            precision_at_i = num_relevant_found / (i + 1)
            ap += precision_at_i

    return ap / len(relevant_doc_ids) if relevant_doc_ids else 0.0

def calculate_mrr(retrieved_doc_ids: List[str], relevant_doc_ids: Set[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) for a single query."""
    if not relevant_doc_ids: return 0.0 # MRR is 0 if no relevant docs exist

    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in relevant_doc_ids:
            return 1.0 / (i + 1)
    return 0.0 # Return 0 if no relevant doc is found in the retrieved list

# --- Evaluation Runner ---
def evaluate_retriever(
    retriever: BaseRetriever,
    queries_dict: Dict[str, str],
    relevance_dict: Dict[str, List[str]],
    k_values: List[int],
    all_documents: Dict[str, str], # Need all docs for re-ranking context
    config: Dict # Pass config for re-ranker settings
) -> Dict:
    """Evaluate a retriever, optionally using a re-ranker based on config."""
    adv_config = config.get('advanced_retrieval', {})
    reranker_enabled = adv_config.get('reranker_enabled', False)
    rerank_top_k = adv_config.get('rerank_top_k', 50) # How many initial candidates to fetch
    final_k = max(k_values) # The final number of results needed for metrics

    # Initialize re-ranker only if enabled and needed
    reranker = None
    if reranker_enabled:
        reranker_model_name = adv_config.get('reranker_model')
        if reranker_model_name:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            try:
                # Placeholder initialization - real init loads the model
                reranker = CrossEncoderReRanker(reranker_model_name, device=device)
                logging.info(f"Re-ranker enabled using placeholder model: {reranker_model_name}")
            except Exception as e:
                logging.error(f"Failed to initialize re-ranker {reranker_model_name}: {e}. Disabling re-ranking.")
                reranker = None # Disable if init fails
        else:
            logging.warning("Re-ranking enabled in config, but no 'reranker_model' specified. Skipping re-ranking.")
            reranker_enabled = False # Disable

    results = {
        'precision': {k: [] for k in k_values},
        'recall': {k: [] for k in k_values},
        'ndcg': {k: [] for k in k_values},
        'map': [],
        'mrr': []
    }

    query_texts_to_eval = list(relevance_dict.keys())
    if not query_texts_to_eval:
        logging.warning("No queries with relevance judgments found. Cannot evaluate.")
        return { # Return zeroed results
            'precision': {k: 0.0 for k in k_values},
            'recall': {k: 0.0 for k in k_values},
            'ndcg': {k: 0.0 for k in k_values},
            'map': 0.0,
            'mrr': 0.0
        }

    logging.info(f"Evaluating {len(query_texts_to_eval)} queries with relevance data...")
    for query_text in tqdm(query_texts_to_eval, desc="Evaluating queries"):
        if reranker_enabled and reranker:
            # --- Re-ranking Pipeline ---
            # 1. Retrieve initial candidates (more than final_k)
            initial_candidates = retriever.retrieve(query_text, rerank_top_k)
            if not initial_candidates:
                retrieved_doc_ids = []
            else:
                # 2. Prepare documents for re-ranker
                candidate_ids = [doc_id for doc_id, _ in initial_candidates]
                docs_to_rerank = {doc_id: all_documents.get(doc_id, "") for doc_id in candidate_ids}
                # Filter out any potential misses from all_documents
                docs_to_rerank = {k: v for k, v in docs_to_rerank.items() if v}

                # 3. Re-rank using the placeholder/real re-ranker
                reranked_results = reranker.rerank(query_text, docs_to_rerank, top_k=final_k)
                retrieved_doc_ids = [doc_id for doc_id, _ in reranked_results]
        else:
            # --- Standard Retrieval Pipeline ---
            retrieved_results = retriever.retrieve(query_text, final_k)
            retrieved_doc_ids = [doc_id for doc_id, _ in retrieved_results]

        # Get the set of relevant document IDs for this query text
        relevant_doc_ids = set(relevance_dict.get(query_text, []))

        # Calculate metrics for each k
        for k in k_values:
            results['precision'][k].append(calculate_precision_at_k(retrieved_doc_ids, relevant_doc_ids, k))
            results['recall'][k].append(calculate_recall_at_k(retrieved_doc_ids, relevant_doc_ids, k))
            results['ndcg'][k].append(calculate_ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k))

        # Calculate MAP and MRR (using the full retrieved list up to max_k)
        results['map'].append(calculate_map(retrieved_doc_ids, relevant_doc_ids))
        results['mrr'].append(calculate_mrr(retrieved_doc_ids, relevant_doc_ids))

    # Calculate averages, handling potential empty lists if no queries were evaluated
    averaged_results = {
        'precision': {k: np.mean(values) if values else 0.0 for k, values in results['precision'].items()},
        'recall': {k: np.mean(values) if values else 0.0 for k, values in results['recall'].items()},
        'ndcg': {k: np.mean(values) if values else 0.0 for k, values in results['ndcg'].items()},
        'map': np.mean(results['map']) if results['map'] else 0.0,
        'mrr': np.mean(results['mrr']) if results['mrr'] else 0.0
    }

    return averaged_results

def initialize_retrievers(config: Dict, documents: Dict[str,str]) -> Dict[str, BaseRetriever]:
    """Initialize retriever models based on the configuration."""
    retriever_config = config['retrieval']
    retrievers = {}

    # --- BM25 ---
    # Assuming BM25 is always included unless explicitly excluded?
    # Or add a flag in config? For now, assume always add.
    logging.info("Initializing BM25 Retriever...")
    retrievers['BM25'] = BM25Retriever()

    # --- Dense Models ---
    dense_models_to_init = {
       f"Dense ({retriever_config.get('default_dense_model')})": DenseRetriever(retriever_config.get('default_dense_model')),
       f"Dense ({retriever_config.get('clinical_dense_model')})": ClinicalDenseRetriever(retriever_config.get('clinical_dense_model')),
       # Add others as needed based on config flags or presence of model names
       # Example: Check if the key exists before adding
       # f"Dense (BioBERT)": BioBERTDenseRetriever(retriever_config.get('biobert_dense_model')),
       # f"Dense (PubMedBERT)": PubMedBERTDenseRetriever(retriever_config.get('pubmedbert_dense_model'))
    }
    # Only initialize models specified in config (check for key presence)
    if retriever_config.get('default_dense_model'):
        logging.info(f"Initializing Dense Retriever ({retriever_config['default_dense_model']})...")
        retrievers[f"Dense (Default)"] = DenseRetriever(retriever_config['default_dense_model'])
    if retriever_config.get('clinical_dense_model'):
        logging.info(f"Initializing Dense Retriever ({retriever_config['clinical_dense_model']})...")
        retrievers[f"Dense (Clinical)"] = ClinicalDenseRetriever(retriever_config['clinical_dense_model'])
    if retriever_config.get('medical_dense_model'):
         logging.info(f"Initializing Dense Retriever ({retriever_config['medical_dense_model']})...")
         retrievers[f"Dense (Medical)"] = MedicalDenseRetriever(retriever_config['medical_dense_model'])
    # Add BioBERT, PubMedBERT similarly if keys exist in config

    # --- Hybrid Models ---
    hybrid_alphas = retriever_config.get('hybrid_alpha_values', [])
    default_dense_for_hybrid = retriever_config.get('default_dense_model')
    if hybrid_alphas and default_dense_for_hybrid:
        for alpha in hybrid_alphas:
            name = f"Hybrid (a={alpha:.1f}) - Default Dense"
            logging.info(f"Initializing {name}...")
            retrievers[name] = HybridRetriever(alpha=alpha, dense_model_name=default_dense_for_hybrid)
    # Optionally allow specifying other dense models for hybrid in config

    # Index all initialized retrievers
    if not documents:
        logging.error("Cannot index retrievers: No documents provided.")
        return {}

    doc_ids = list(documents.keys())
    doc_texts = list(documents.values())

    for name, retriever in retrievers.items():
        logging.info(f"\nIndexing documents for {name}...")
        try:
            retriever.index(doc_texts, doc_ids)
        except Exception as e:
            logging.error(f"Failed to index retriever {name}: {e}")
            # Optionally remove the failed retriever?
            # del retrievers[name]

    return retrievers

def main(config: Dict):
    """Load data, initialize and evaluate retrievers based on config."""
    path_config = config['paths']
    eval_config = config['evaluation']

    # --- Load Data ---
    # Determine which dataset(s) to load from config
    # Example: Load final pubmed set and potentially the medical set
    datasets_to_load = []
    if path_config.get('pubmed_dataset_final'):
        datasets_to_load.append(path_config['pubmed_dataset_final'])
    if path_config.get('medical_dataset'): # Check if medical_dataset path exists
        # Decide whether to combine or use separately - current code combines
        datasets_to_load.append(path_config['medical_dataset'])

    if not datasets_to_load:
        logging.error("No dataset file paths specified in config (e.g., 'pubmed_dataset_final'). Exiting.")
        return

    all_documents = {}
    all_queries = {}
    all_relevance = {}

    logging.info(f"Loading datasets: {datasets_to_load}")
    for file_path in datasets_to_load:
        try:
            dataset = load_dataset(file_path)
            # Merge dictionaries, handling potential key collisions if necessary
            # Simple update assumes unique IDs across files or overwriting is acceptable
            all_documents.update(dataset.get('documents', {}))
            all_queries.update(dataset.get('queries', {}))
            all_relevance.update(dataset.get('relevance', {}))
            logging.info(f"Loaded {len(dataset.get('documents', {}))} docs, {len(dataset.get('queries', {}))} queries from {file_path}")
        except Exception as e:
            logging.error(f"Failed to load or process dataset {file_path}: {e}. Skipping this file.")
            continue

    if not all_documents or not all_queries:
        logging.error("No documents or queries loaded successfully. Cannot proceed with evaluation.")
        return

    logging.info(f"Total unique documents loaded: {len(all_documents)}")
    logging.info(f"Total unique queries loaded: {len(all_queries)}")
    logging.info(f"Total query-relevance pairs loaded: {len(all_relevance)}")

    # --- Initialize Retrievers --- (includes indexing)
    retrievers = initialize_retrievers(config, all_documents)
    if not retrievers:
         logging.error("No retrievers were initialized successfully. Exiting.")
         return

    # --- Evaluate Retrievers ---
    k_values = eval_config.get('k_values', [1, 5, 10]) # Default k values
    results = {}

    # We need the query texts for evaluation, and relevance is keyed by query text
    # The evaluate_retriever function expects the relevance dict {query_text: [doc_id,...]}
    queries_for_eval = {qid: qtext for qid, qtext in all_queries.items() if qtext in all_relevance}
    if not queries_for_eval:
        logging.error("No queries found that have corresponding relevance judgments. Check dataset alignment.")
        return

    logging.info(f"Starting evaluation for {len(retrievers)} retriever(s) on {len(queries_for_eval)} queries with relevance data.")

    for name, retriever in retrievers.items():
        logging.info(f"\n--- Evaluating {name} ---")
        # Pass all_documents and config to the evaluation function
        results[name] = evaluate_retriever(
            retriever, all_queries, all_relevance, k_values, all_documents, config
        )

    # --- Print and Save Results ---
    logging.info("\n--- Evaluation Results Summary ---")
    print("=" * 80) # Use print for final summary table

    for name, metrics in results.items():
        print(f"\n{name}:")
        print("-" * 40)
        for metric in ['precision', 'recall', 'ndcg']:
            print(f"  {metric.upper()}:")
            for k in k_values:
                print(f"    @{k}: {metrics[metric][k]:.4f}")
        print(f"  MAP: {metrics['map']:.4f}")
        print(f"  MRR: {metrics['mrr']:.4f}")
    print("=" * 80)

    # Save results to file specified in config
    output_file = path_config.get('evaluation_results', 'evaluation_results.json')
    logging.info(f"Saving detailed results to {output_file}...")
    try:
        with open(output_file, "w", encoding='utf-8') as f:
            # Convert numpy types to native Python types for JSON serialization
            serializable_results = json.loads(json.dumps(results, default=lambda x: x.item() if isinstance(x, np.generic) else x))
            json.dump(serializable_results, f, indent=4)
        logging.info(f"Results saved to {output_file}")
    except IOError as e:
        logging.error(f"Error writing results to file {output_file}: {e}")
    except TypeError as e:
        logging.error(f"Error serializing results to JSON: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate medical information retrieval models.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file (default: config.yaml)"
    )
    args = parser.parse_args()

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {args.config}")
        exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file {args.config}: {e}")
        exit(1)
    except Exception as e:
         logging.error(f"An unexpected error occurred loading config: {e}")
         exit(1)

    main(config)
    logging.info("Evaluation process finished.") 