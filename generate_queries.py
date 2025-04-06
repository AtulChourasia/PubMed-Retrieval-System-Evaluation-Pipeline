import json
from typing import List, Dict, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
import yaml
import logging
import random
import numpy as np
from fetch_pubmed import call_llm_for_relevance # Reuse relevance judgment LLM call

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Constants (can be overridden by config file) ---
# These provide defaults if not found in the config, but loading from config is preferred.
DEFAULT_STOP_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'by'
}

# --- Placeholder for LLM Interaction (If needed directly here, otherwise rely on fetch_pubmed) ---
# Potentially reuse the one from fetch_pubmed if called from there, or define/import here if needed.
from fetch_pubmed import call_llm_for_relevance # Reuse relevance judgment LLM call

# --- Placeholder for LLM Query Generation (Requires User Implementation) ---
def call_llm_for_query_generation(doc_text: str, config: Dict) -> List[str]:
    """Placeholder function to call an LLM API for query generation.

    Args:
        doc_text: The document text (title + abstract).
        config: The loaded configuration dictionary.

    Returns:
        A list of generated query strings or an empty list if the call fails.
    """
    llm_config = config.get('llm', {})
    query_gen_config = llm_config.get('query_generation', {})
    prompt_template = query_gen_config.get('prompt_template', "Abstract: {abstract}\nGenerate 3 questions:")
    max_queries = query_gen_config.get('max_queries_per_doc', 3)

    # !!! USER IMPLEMENTATION NEEDED HERE !!!
    # Similar to call_llm_for_relevance:
    # 1. Get API key
    # 2. Format prompt (extract abstract if needed)
    #    prompt = prompt_template.format(abstract=doc_text) # Adjust formatting
    # 3. Make API call using appropriate library and model
    # 4. Parse response into a list of strings (one per generated query)
    # 5. Handle errors, rate limits, etc.
    # 6. Return the list of query strings (up to max_queries).

    logging.warning("LLM query generation not implemented. Returning empty list.")
    # Example structure:
    # try:
    #     # ... make API call ...
    #     response_text = ...
    #     queries = [q.strip() for q in response_text.split('\n') if q.strip()]
    #     return queries[:max_queries]
    # except Exception as e:
    #     logging.error(f"LLM query generation API call failed: {e}")
    #     return []
    return []

# --- Utility Functions ---
def load_dataset(file_path: str) -> Dict:
    """Load the dataset from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            # Basic validation of expected structure
            if not all(k in dataset for k in ['documents', 'queries', 'relevance']):
                logging.error(f"Dataset file {file_path} is missing required keys (documents, queries, relevance).")
                raise ValueError("Invalid dataset format")
            if not isinstance(dataset['documents'], dict):
                 logging.warning(f"Expected 'documents' to be a dict in {file_path}, converting from list if possible...")
                 # Attempt conversion if it's a list of strings (old format)
                 if isinstance(dataset['documents'], list) and all(isinstance(d, str) for d in dataset['documents']):
                     dataset['documents'] = {f"doc_{i}": doc for i, doc in enumerate(dataset['documents'])}
                 else:
                     raise ValueError("Invalid format for 'documents'. Expected dict {doc_id: text}.")
            if not isinstance(dataset['queries'], dict):
                 logging.warning(f"Expected 'queries' to be a dict in {file_path}, converting from list if possible...")
                 # Attempt conversion if it's a list of strings (old format)
                 if isinstance(dataset['queries'], list) and all(isinstance(q, str) for q in dataset['queries']):
                     dataset['queries'] = {f"Q_{i+1}": q for i, q in enumerate(dataset['queries'])}
                 else:
                     raise ValueError("Invalid format for 'queries'. Expected dict {query_id: text}.")

            return dataset
    except FileNotFoundError:
        logging.error(f"Dataset file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from {file_path}: {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading dataset from {file_path}: {e}")
        raise

def clean_topic(topic: str, config: Dict) -> str:
    """Clean and validate a topic using configuration settings."""
    cfg = config['query_generation']['topic_cleaning']
    stop_words = set(cfg.get('stop_words', DEFAULT_STOP_WORDS))
    min_meaningful_words = cfg.get('min_meaningful_words', 2)
    max_topic_words = cfg.get('max_topic_words', 5)

    words = topic.lower().split()
    cleaned = [w for w in words if w not in stop_words]

    if len(cleaned) < min_meaningful_words:
        return ""

    cleaned_topic = " ".join(cleaned)
    cleaned_topic = cleaned_topic.strip('.,:;?!') # More comprehensive stripping

    # Re-check length after joining and stripping
    final_words = cleaned_topic.split()
    if not (min_meaningful_words <= len(final_words) <= max_topic_words):
        return ""

    return cleaned_topic

def is_valid_medical_topic(topic: str, config: Dict) -> bool:
    """Check if a topic is a valid medical topic based on configuration."""
    cfg = config['query_generation']['topic_validation']
    medical_terms_config = cfg.get('medical_terms', {}) # Load from config
    min_words = cfg.get('min_words', 2)
    max_words = cfg.get('max_words', 5)
    # Consider adding invalid endings/beginnings check from config if desired

    words = topic.split() # Assumes topic is already reasonably cleaned
    if not (min_words <= len(words) <= max_words):
        return False

    # Check for presence of configured medical terms
    has_medical_term = False
    for category, terms in medical_terms_config.items():
        if any(term.lower() in topic.lower() for term in terms):
            has_medical_term = True
            break

    if not has_medical_term:
        # logging.debug(f"Topic '{topic}' rejected: No medical term found.")
        return False

    # TODO: Add checks for incomplete endings/beginnings if needed, loading from config

    return True

def extract_topics_from_documents(documents: Dict[str, str], config: Dict) -> Set[str]:
    """Extract meaningful topics from document texts using configuration."""
    cfg_extract = config['query_generation']['topic_extraction']
    num_sentences = cfg_extract.get('num_sentences_to_check', 3)

    topics = set()
    logging.info(f"Extracting topics from {len(documents)} documents...")
    # Use document values (texts) for extraction
    for doc_text in tqdm(list(documents.values()), desc="Extracting topics"):
        # Basic sentence splitting (consider NLTK/SpaCy for robustness)
        sentences = doc_text.split('.')
        processed_sentences = 0
        for sentence in sentences:
            if not sentence.strip(): continue # Skip empty sentences
            if processed_sentences >= num_sentences: break

            words = sentence.lower().split()
            if len(words) >= 3:
                 # Try different phrase lengths (configurable? currently 2-4)
                for length in [4, 3, 2]:
                    for i in range(len(words) - length + 1):
                        phrase = " ".join(words[i:i+length])
                        cleaned_topic = clean_topic(phrase, config)
                        if cleaned_topic and is_valid_medical_topic(cleaned_topic, config):
                            if cleaned_topic not in topics:
                                topics.add(cleaned_topic)
                                # Optional: Break inner loop once a valid topic is found per position?
            processed_sentences += 1

    logging.info(f"Extracted {len(topics)} unique candidate topics.")
    return topics

def generate_additional_queries(topics: Set[str], existing_queries: Dict[str, str], config: Dict) -> List[str]:
    """Generate additional queries from topics OR LLM based on config."""
    llm_config = config.get('llm', {})
    use_llm = llm_config.get('enabled_query_generation', False)

    if use_llm:
        logging.warning("LLM-based query generation from topics is not implemented yet. Falling back to template method.")
        # TODO: Implement logic to feed topics/context to LLM for query generation if desired
        # Fallback to template method for now:
        return _generate_queries_from_templates(topics, existing_queries, config)
    else:
        logging.info("Generating additional queries using templates...")
        return _generate_queries_from_templates(topics, existing_queries, config)

def _generate_queries_from_templates(topics: Set[str], existing_queries: Dict[str, str], config: Dict) -> List[str]:
    """Helper function for template-based query generation (original logic)."""
    cfg_gen_topics = config['query_generation']['query_generation_from_topics']
    # Use dataset_generation templates as they are likely broader
    query_templates = config['dataset_generation'].get('query_templates', [])
    num_templates_per_topic = cfg_gen_topics.get('num_templates_per_topic', 3)
    top_n_topics = cfg_gen_topics.get('top_n_topics', 500)

    sorted_topics = sorted(list(topics), key=len, reverse=True)[:top_n_topics]
    logging.info(f"Generating queries from top {len(sorted_topics)} topics using templates...")

    new_queries = set()
    existing_query_texts = set(existing_queries.values())

    if not query_templates:
        logging.warning("No query templates found in configuration. Cannot generate additional queries.")
        return []

    for topic in tqdm(sorted_topics, desc="Generating queries from topics"):
        # Safely sample templates
        k = min(num_templates_per_topic, len(query_templates))
        templates_to_sample = random.sample(query_templates, k)
        if k < num_templates_per_topic:
             logging.warning(f"Requested {num_templates_per_topic} templates, but only {k} available.")

        for template in templates_to_sample:
            try:
                query = template.format(topic)
                if query not in existing_query_texts and query not in new_queries:
                    new_queries.add(query)
            except (KeyError, ValueError, IndexError) as e:
                 logging.warning(f"Template '{template}' incompatible with topic '{topic}'. Error: {e}")

    logging.info(f"Generated {len(new_queries)} new unique queries from topics.")
    return list(new_queries)

def update_relevance_judgments(documents: Dict[str, str], queries: Dict[str, str], existing_relevance: Dict[str, List[str]], config: Dict) -> Dict[str, List[str]]:
    """Update relevance judgments using TF-IDF or LLM based on config. Handles new queries."""
    cfg_relevance = config['query_generation']['relevance_update']
    llm_config = config.get('llm', {})
    relevance_llm_config = llm_config.get('relevance', {})
    use_llm = llm_config.get('enabled_relevance', False)

    if not documents:
        logging.warning("No documents provided for relevance update.")
        return existing_relevance

    doc_ids = list(documents.keys())
    doc_texts = list(documents.values())
    updated_relevance = existing_relevance.copy()
    all_query_texts = list(queries.values()) # List of all query texts (existing + new)

    # Determine which query texts actually need judgments (are new)
    queries_to_judge = [q_text for q_text in all_query_texts if q_text not in updated_relevance]
    logging.info(f"Found {len(queries_to_judge)} queries needing relevance judgments.")

    if not queries_to_judge:
        return updated_relevance # No new queries to judge

    if use_llm:
        logging.info(f"Updating relevance judgments for {len(queries_to_judge)} queries using LLM (placeholder)...")
        max_pairs = relevance_llm_config.get('max_pairs_per_query', 50)
        # Requires TF-IDF for candidate sampling
        logging.info("Calculating TF-IDF vectors for LLM candidate sampling...")
        tfidf = TfidfVectorizer(stop_words='english')
        try:
            doc_vectors = tfidf.fit_transform(doc_texts)
        except ValueError as e:
            logging.error(f"TF-IDF Error for LLM sampling in relevance update: {e}. Cannot update LLM judgments.")
            return updated_relevance # Return original relevance

        for query_text in tqdm(queries_to_judge, desc="Updating relevance via LLM (placeholder)"):
            if query_text in updated_relevance: continue # Should not happen based on queries_to_judge, but double-check
            try:
                query_vector = tfidf.transform([query_text])
            except ValueError:
                logging.warning(f"Query '{query_text}' could not be transformed for LLM sampling. Skipping.")
                continue

            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            candidate_indices = np.argsort(similarities)[-max_pairs:][::-1]

            llm_relevant_docs = []
            for idx in candidate_indices:
                doc_id = doc_ids[idx]
                doc_text = documents[doc_id]
                # --- LLM Call Placeholder ---
                llm_score = call_llm_for_relevance(query_text, doc_text, config)
                # --- End LLM Call Placeholder ---
                if llm_score is not None and llm_score >= 2: # Example threshold
                    llm_relevant_docs.append(doc_id)

            if llm_relevant_docs:
                updated_relevance[query_text] = llm_relevant_docs

    else: # Use TF-IDF
        logging.info(f"Updating relevance judgments for {len(queries_to_judge)} queries using TF-IDF...")
        tfidf_threshold = cfg_relevance.get('tfidf_threshold', 0.1)
        top_n_matches = cfg_relevance.get('top_n_matches', 20)

        tfidf = TfidfVectorizer(stop_words='english')
        try:
            doc_vectors = tfidf.fit_transform(doc_texts)
        except ValueError as e:
            logging.error(f"TF-IDF Error during relevance update: {e}. Skipping TF-IDF update.")
            return updated_relevance # Return original relevance

        for query_text in tqdm(queries_to_judge, desc="Updating relevance via TF-IDF"):
            if query_text in updated_relevance: continue
            try:
                query_vector = tfidf.transform([query_text])
            except ValueError:
                 logging.warning(f"Query '{query_text}' could not be transformed for TF-IDF relevance. Skipping.")
                 continue

            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            effective_top_n = min(top_n_matches, len(doc_ids))
            top_indices = np.argsort(similarities)[-effective_top_n:][::-1]

            relevant_docs_for_query = []
            for idx in top_indices:
                if similarities[idx] > tfidf_threshold:
                    relevant_docs_for_query.append(doc_ids[idx])

            if relevant_docs_for_query:
                updated_relevance[query_text] = relevant_docs_for_query

    logging.info(f"Relevance judgments update complete. Total queries with relevance: {len(updated_relevance)}")
    return updated_relevance

def main(config: Dict):
    """Main function to generate queries and update dataset using config."""
    path_config = config['paths']
    input_dataset_path = path_config['pubmed_dataset']
    output_dataset_path = path_config['pubmed_dataset_expanded']

    # Set random seed for reproducibility if specified
    if config.get('random_seed'):
        seed = int(config['random_seed'])
        random.seed(seed)
        logging.info(f"Set random seed to {seed}")

    # Load existing dataset
    logging.info(f"Loading existing dataset from {input_dataset_path}...")
    try:
        dataset = load_dataset(input_dataset_path)
    except Exception:
        logging.error("Failed to load initial dataset. Exiting.")
        return

    documents_dict = dataset['documents']
    queries_dict = dataset['queries']
    relevance_dict = dataset['relevance']

    logging.info(f"Current dataset stats: Documents: {len(documents_dict)}, Queries: {len(queries_dict)}")

    # Extract topics from existing documents
    topics = extract_topics_from_documents(documents_dict, config)

    # Generate additional queries
    logging.info("Generating additional queries...")
    new_query_texts = generate_additional_queries(topics, queries_dict, config)

    # Combine existing and new queries
    # Create new query IDs for the new queries
    max_existing_q_id = 0
    for q_id in queries_dict.keys():
        if q_id.startswith('Q_') and q_id[2:].isdigit():
            max_existing_q_id = max(max_existing_q_id, int(q_id[2:]))

    all_queries_dict = queries_dict.copy()
    for i, new_q_text in enumerate(new_query_texts):
        new_q_id = f"Q_{max_existing_q_id + i + 1}"
        all_queries_dict[new_q_id] = new_q_text

    logging.info(f"Total queries after generation: {len(all_queries_dict)}")

    # Update relevance judgments for all (including new) queries
    # Pass the combined query dictionary
    logging.info("Updating relevance judgments...")
    updated_relevance = update_relevance_judgments(
        documents_dict,
        all_queries_dict,
        relevance_dict,
        config
    )

    # Create updated dataset structure
    updated_dataset = {
        "documents": documents_dict,
        "queries": all_queries_dict,
        "relevance": updated_relevance
    }

    # Save updated dataset
    logging.info(f"Saving updated dataset to {output_dataset_path}...")
    try:
        with open(output_dataset_path, "w", encoding="utf-8") as f:
            json.dump(updated_dataset, f, indent=4, ensure_ascii=False)
        logging.info("Dataset saved successfully.")
    except IOError as e:
        logging.error(f"Error writing updated dataset to {output_dataset_path}: {e}")
    except TypeError as e:
        logging.error(f"Error serializing updated dataset to JSON: {e}")

    logging.info(f"\nFinal dataset stats:")
    logging.info(f"Documents: {len(updated_dataset['documents'])}")
    logging.info(f"Total Queries: {len(updated_dataset['queries'])}")
    logging.info(f"New queries added: {len(new_query_texts)}")

    # Print some example new queries
    if new_query_texts:
        logging.info("\nExample newly generated queries:")
        logging.info("-" * 40)
        for query in new_query_texts[:10]: # Log first 10 new queries
            logging.info(query)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate additional queries for a medical retrieval dataset.")
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

    main(config)
    logging.info("Query generation process finished.") 