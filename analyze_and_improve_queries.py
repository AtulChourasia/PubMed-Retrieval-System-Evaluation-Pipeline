import json
from collections import Counter
from typing import List, Dict, Set
from tqdm import tqdm
import re
import argparse
import yaml
import logging
import random

# Assuming update_relevance_judgments is defined in generate_queries
# This might be better placed in a shared utils module
from generate_queries import update_relevance_judgments, load_dataset as load_base_dataset
# Import the LLM query generation placeholder
from generate_queries import call_llm_for_query_generation

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Constants (Defaults) ---
# These provide defaults if not found in the config.
DEFAULT_MEDICAL_AFFIXES = ['cardio', 'neuro', 'gastro', 'hepat', 'onco', 'patho', 'itis', 'osis', 'emia', 'oma', 'pathy', 'plasty']
DEFAULT_INVALID_ENDINGS = ['and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'by', 'for', 'to', 'with', 'early', 'late', 'severe', 'mild', 'acute', 'chronic', 'clinical', 'medical']
DEFAULT_CONCEPT_PATTERNS = [
    r'\b\w+(?:itis|osis|emia|oma|pathy)\b',
    r'\b(?:acute|chronic)\s+\w+(?:\s+\w+)?\s+(?:disease|syndrome|disorder)\b',
    r'\b\w+(?:\s+\w+)?\s+(?:therapy|treatment|surgery|procedure)\b',
]
DEFAULT_IMPROVED_TEMPLATES = [
    "What is the current standard of care for {}?",
    "What are the evidence-based guidelines for managing {}?",
]

# --- Utility Functions ---
def load_dataset(file_path: str) -> Dict:
    """Load the dataset, reusing the base loader and adding checks if needed."""
    # We can potentially add more specific checks for this script if necessary
    # For now, reusing the loader from generate_queries which handles format conversion
    logging.info(f"Loading dataset using base loader from: {file_path}")
    return load_base_dataset(file_path)

def analyze_queries(queries: Dict[str, str], config: Dict) -> Dict:
    """Analyze existing queries (provided as dict) for patterns and topics using config."""
    cfg_analyze = config.get('query_analysis', {}) # Optional section in config
    medical_affixes = cfg_analyze.get('medical_affixes', DEFAULT_MEDICAL_AFFIXES)

    question_types = Counter()
    medical_terms_found = Counter() # Renamed for clarity
    query_lengths = Counter()

    logging.info(f"Analyzing {len(queries)} queries...")
    for query_text in tqdm(list(queries.values()), desc="Analyzing queries"):
        lower_query = query_text.lower()
        # Basic question type analysis (can be improved with regex/NLP)
        if "how" in lower_query:
            question_types['how'] += 1
        elif "what" in lower_query:
            question_types['what'] += 1
        elif "when" in lower_query:
            question_types['when'] += 1
        elif "why" in lower_query:
            question_types['why'] += 1
        # Add more types? (Where, Which, etc.)

        words = query_text.split()
        query_lengths[len(words)] += 1

        # Find potential medical terms based on affixes (simple heuristic)
        for word in words:
            word_lower = word.lower().strip('.,:;?!')
            if any(affix in word_lower for affix in medical_affixes):
                medical_terms_found[word_lower] += 1

    analysis = {
        'question_types': dict(question_types),
        'potential_medical_terms_by_affix': dict(medical_terms_found),
        'query_lengths': dict(query_lengths)
    }
    logging.info("Query analysis complete.")
    return analysis

def is_valid_medical_concept(concept: str, config: Dict) -> bool:
    """Validate if a concept is a proper medical term based on config."""
    # Use the consolidated config section
    cfg = config['query_improvement']['topic_validation']
    min_words = cfg.get('min_words', 2)
    invalid_endings = cfg.get('invalid_endings', DEFAULT_INVALID_ENDINGS)
    medical_terms_dict = cfg.get('medical_terms', {}) # Load medical terms dict

    words = concept.split()
    if len(words) < min_words:
        return False

    # Check invalid endings/beginnings
    # TODO: Add check for invalid start words from config?
    if any(concept.endswith(f" {word}") for word in invalid_endings):
        return False

    # Check for presence of required medical terms
    has_medical_term = False
    for category_terms in medical_terms_dict.values():
        if any(term.lower() in concept.lower() for term in category_terms):
            has_medical_term = True
            break

    return has_medical_term

def extract_medical_concepts(text: str, config: Dict) -> Set[str]:
    """Extract medical concepts from text using regex patterns from config."""
    cfg_extract = config['query_improvement']['concept_extraction']
    patterns = cfg_extract.get('patterns', DEFAULT_CONCEPT_PATTERNS)

    concepts = set()
    for pattern in patterns:
        try:
            # Case-insensitive matching
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                concept_text = match.group(0).lower().strip()
                # Validate the extracted concept using config
                if is_valid_medical_concept(concept_text, config):
                    concepts.add(concept_text)
        except re.error as e:
            logging.warning(f"Invalid regex pattern skipped: {pattern}. Error: {e}")
            continue

    return concepts

def generate_improved_queries(documents: Dict[str, str], existing_queries: Dict[str, str], config: Dict) -> List[str]:
    """Generate improved queries based on document analysis (concepts/regex) OR LLM using config."""
    llm_config = config.get('llm', {})
    use_llm = llm_config.get('enabled_query_generation', False)

    if use_llm:
        logging.info("Generating improved queries using LLM (placeholder)...")
        return _generate_queries_via_llm(documents, existing_queries, config)
    else:
        logging.info("Generating improved queries using concept extraction and templates...")
        return _generate_queries_via_concepts(documents, existing_queries, config)

def _generate_queries_via_llm(documents: Dict[str, str], existing_queries: Dict[str, str], config: Dict) -> List[str]:
    """Helper function for LLM-based query generation."""
    new_queries = set()
    existing_query_texts = set(existing_queries.values())

    logging.info(f"Generating queries via LLM for {len(documents)} documents (placeholder)... ")
    for doc_id, doc_text in tqdm(documents.items(), desc="Generating queries via LLM"):
        # --- LLM Call Placeholder ---
        generated_qs = call_llm_for_query_generation(doc_text, config)
        # --- End LLM Call Placeholder ---

        for query in generated_qs:
            if query not in existing_query_texts and query not in new_queries:
                new_queries.add(query)

    logging.info(f"Generated {len(new_queries)} new unique queries via LLM (placeholder). ")
    return list(new_queries)

def _generate_queries_via_concepts(documents: Dict[str, str], existing_queries: Dict[str, str], config: Dict) -> List[str]:
    """Helper function for concept/template-based query generation (original logic)."""
    cfg_gen = config['query_improvement']['query_generation_from_concepts']
    improved_templates = cfg_gen.get('improved_templates', DEFAULT_IMPROVED_TEMPLATES)
    num_templates_per_concept = cfg_gen.get('num_templates_per_concept', 5)
    top_n_concepts = cfg_gen.get('top_n_concepts', 300)

    if not improved_templates:
        logging.warning("No improved query templates found in config. Cannot generate queries.")
        return []

    all_concepts = set()
    logging.info(f"Extracting medical concepts from {len(documents)} documents...")
    for doc_text in tqdm(list(documents.values()), desc="Extracting concepts"):
        concepts = extract_medical_concepts(doc_text, config)
        all_concepts.update(concepts)
    logging.info(f"Extracted {len(all_concepts)} unique candidate concepts.")

    if not all_concepts:
         logging.warning("No concepts extracted. Cannot generate queries via concepts. Check config patterns/validation.")
         return []

    sorted_concepts = sorted(list(all_concepts), key=lambda x: (-len(x.split()), x))[:top_n_concepts]
    logging.info(f"Generating improved queries from top {len(sorted_concepts)} concepts...")

    new_queries = set()
    existing_query_texts = set(existing_queries.values())

    for concept in tqdm(sorted_concepts, desc="Generating improved queries from concepts"):
        k = min(num_templates_per_concept, len(improved_templates))
        templates_to_sample = random.sample(improved_templates, k)
        if k < num_templates_per_concept:
            logging.warning(f"Requested {num_templates_per_concept} templates, but only {k} available.")

        for template in templates_to_sample:
            try:
                query = template.format(concept)
                if query not in existing_query_texts and query not in new_queries:
                    new_queries.add(query)
            except (KeyError, ValueError, IndexError) as e:
                 logging.warning(f"Template '{template}' likely incompatible with concept '{concept}'. Error: {e}")

    logging.info(f"Generated {len(new_queries)} new unique improved queries via concepts.")
    return list(new_queries)

def main(config: Dict):
    """Main function to analyze, improve queries, and update dataset using config."""
    path_config = config['paths']
    # Input is the expanded dataset from generate_queries.py
    input_dataset_path = path_config['pubmed_dataset_expanded']
    # Output is the final dataset
    output_dataset_path = path_config['pubmed_dataset_final']

    # Set random seed if specified
    if config.get('random_seed'):
        seed = int(config['random_seed'])
        random.seed(seed)
        logging.info(f"Set random seed to {seed}")

    # Load dataset
    logging.info(f"Loading dataset from {input_dataset_path}...")
    try:
        dataset = load_dataset(input_dataset_path)
    except Exception:
        logging.error("Failed to load dataset for analysis/improvement. Exiting.")
        return

    documents_dict = dataset['documents']
    queries_dict = dataset['queries']
    relevance_dict = dataset['relevance']

    # Analyze existing queries
    logging.info("Analyzing existing queries...")
    analysis = analyze_queries(queries_dict, config)

    logging.info("\nQuery Analysis Results:")
    logging.info("-" * 40)
    logging.info(f"Question Types: {analysis['question_types']}")
    logging.info(f"Query Lengths: {analysis['query_lengths']}")
    top_n_terms = 10
    top_terms = dict(sorted(analysis['potential_medical_terms_by_affix'].items(), key=lambda x: x[1], reverse=True)[:top_n_terms])
    logging.info(f"Top {top_n_terms} Potential Medical Terms (by affix): {top_terms}")

    # Generate improved queries based on document concepts
    logging.info("\nGenerating improved queries based on document concepts...")
    new_improved_query_texts = generate_improved_queries(documents_dict, queries_dict, config)

    # Update dataset: Add new queries
    max_existing_q_id = 0
    for q_id in queries_dict.keys():
        if q_id.startswith('Q_') and q_id[2:].isdigit():
            max_existing_q_id = max(max_existing_q_id, int(q_id[2:]))

    all_queries_dict = queries_dict.copy()
    num_new_queries_added = 0
    for i, new_q_text in enumerate(new_improved_query_texts):
        new_q_id = f"Q_{max_existing_q_id + i + 1}"
        # Ensure the text isn't somehow already present (should be handled by generation func)
        if new_q_text not in queries_dict.values():
             all_queries_dict[new_q_id] = new_q_text
             num_new_queries_added += 1

    logging.info(f"Added {num_new_queries_added} unique new queries. Total queries: {len(all_queries_dict)}")

    # Update relevance judgments for the combined set of queries
    logging.info("\nUpdating relevance judgments for the final query set...")
    # Pass the final combined query dictionary
    # Re-use the function imported from generate_queries
    updated_relevance = update_relevance_judgments(
        documents_dict,
        all_queries_dict,
        relevance_dict, # Pass existing relevance to potentially update/add to
        config
    )

    # Create final dataset structure
    final_dataset = {
        "documents": documents_dict,
        "queries": all_queries_dict,
        "relevance": updated_relevance
    }

    # Save final dataset
    logging.info(f"\nSaving final dataset to {output_dataset_path}...")
    try:
        with open(output_dataset_path, "w", encoding="utf-8") as f:
            json.dump(final_dataset, f, indent=4, ensure_ascii=False)
        logging.info("Final dataset saved successfully.")
    except IOError as e:
        logging.error(f"Error writing final dataset to {output_dataset_path}: {e}")
    except TypeError as e:
        logging.error(f"Error serializing final dataset to JSON: {e}")

    logging.info(f"\nFinal Dataset Stats:")
    logging.info(f"Documents: {len(final_dataset['documents'])}")
    logging.info(f"Total Queries: {len(final_dataset['queries'])}")
    logging.info(f"New queries added in this step: {num_new_queries_added}")

    # Print example new queries
    if new_improved_query_texts:
        logging.info("\nExample New Queries Generated in this step:")
        logging.info("-" * 40)
        for query in new_improved_query_texts[:10]: # Log first 10 new queries
            logging.info(query)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze documents and generate improved queries for a medical retrieval dataset.")
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
    logging.info("Query analysis and improvement process finished.") 