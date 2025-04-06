import requests
import json
from typing import List, Dict
import time
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import yaml
import logging
import numpy as np
import os # Added for environment variable access
import google.generativeai as genai # Added for Gemini API
import re # Added for regular expression operations

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- PubMed API Interaction ---
def fetch_pubmed_ids(query: str, config: Dict) -> List[str]:
    """Fetch article IDs from PubMed for a given query."""
    api_config = config['pubmed_api']
    search_url = f"{api_config['base_url']}esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": api_config['max_results_per_topic'],
        "retmode": "json"
    }
    try:
        response = requests.get(search_url, params=search_params, timeout=api_config['search_timeout'])
        response.raise_for_status()
        search_data = response.json()

        if "esearchresult" not in search_data or "idlist" not in search_data["esearchresult"]:
            logging.warning(f"No PubMed IDs found for query: {query}")
            return []
        return search_data["esearchresult"]["idlist"]
    except requests.exceptions.RequestException as e:
        logging.error(f"Error searching PubMed IDs for query '{query}': {str(e)}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON response for PubMed search '{query}': {str(e)}")
        return []

def fetch_pubmed_details(article_ids: List[str], config: Dict) -> List[Dict]:
    """Fetch details (title, abstract) for a list of PubMed IDs."""
    api_config = config['pubmed_api']
    base_url = api_config['base_url']
    articles = []
    batch_size = api_config['batch_size']

    for i in range(0, len(article_ids), batch_size):
        batch_ids = article_ids[i:i + batch_size]
        fetch_url = f"{base_url}esummary.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(batch_ids),
            "retmode": "json"
        }

        try:
            response = requests.get(fetch_url, params=fetch_params, timeout=api_config['fetch_timeout'])
            response.raise_for_status()
            article_data = response.json()

            if "result" in article_data:
                for article_id in batch_ids:
                    if article_id in article_data["result"]:
                        article = article_data["result"][article_id]
                        # Ensure essential fields exist and handle potential missing abstracts
                        title = article.get("title", "")
                        abstract = article.get("abstract", "") # PubMed sometimes lacks abstracts
                        if title: # Only add articles with titles
                           articles.append({
                               "pmid": article_id, # Store PMID for reference
                               "title": title,
                               "abstract": abstract
                           })
            else:
                logging.warning(f"No 'result' key in PubMed summary response for batch starting with ID {batch_ids[0]}")

            time.sleep(api_config['delay_between_batches'])
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching PubMed details batch: {str(e)}")
            logging.info(f"Waiting {api_config['delay_on_error']}s before retrying or skipping...")
            time.sleep(api_config['delay_on_error'])
            continue # Consider more sophisticated retry logic
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON response for PubMed details batch: {str(e)}")
            time.sleep(api_config['delay_on_error'])
            continue

    return articles


# --- LLM Interaction using Google Generative AI ---
def call_llm_for_relevance(query: str, doc_text: str, config: Dict) -> int | None:
    """Calls the Google Generative AI API (Gemini) for relevance judgment,
       using the genai.Client approach.

    Args:
        query: The query text.
        doc_text: The document text (containing title and abstract).
        config: The loaded configuration dictionary.

    Returns:
        An integer relevance score (0-3) or None if the call fails or returns invalid output.
    """
    llm_config = config.get('llm', {})
    relevance_config = llm_config.get('relevance', {})
    api_provider = llm_config.get('api_provider')

    # Ensure the provider is Google for this function
    if api_provider != 'google':
        logging.error(f"API provider is set to '{api_provider}', but expected 'google' for this function.")
        return None

    api_key_var = llm_config.get('api_key_env_var', 'GOOGLE_API_KEY')
    api_key = os.environ.get(api_key_var)

    if not api_key:
        logging.error(f"Google API key not found in environment variable '{api_key_var}'. Cannot call LLM.")
        return None

    try:
        # Initialize client directly using the API key
        client = genai.Client(api_key=api_key)
    except Exception as e:
        logging.error(f"Failed to initialize Google Generative AI Client: {e}")
        return None

    model_name = relevance_config.get('model', 'gemini-1.5-flash-latest')
    prompt_template = relevance_config.get('prompt_template', "Query: {query}\nDocument Abstract: {abstract}\nRelevance Score (0-3):")

    # Extract abstract (simple heuristic)
    parts = doc_text.split('\n', 1)
    abstract = parts[1] if len(parts) > 1 else doc_text
    # title = parts[0] if len(parts) > 1 else "N/A" # Title not used in default prompt

    # Format the prompt
    prompt = prompt_template.format(query=query, abstract=abstract.strip())

    try:
        # Add safety settings (optional but recommended)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        # Configure generation parameters (optional)
        generation_config = genai.types.GenerationConfig(
            candidate_count=1,
            temperature=0.1,
            max_output_tokens=10
        )

        # Use client.models.generate_content
        # Note: Model name needs to be prefixed with "models/" when using client
        response = client.generate_content(
            model=f"models/{model_name}", # Prefix model name
            contents=prompt, # Use 'contents' argument
            generation_config=generation_config,
            safety_settings=safety_settings
            )

        # Extract and validate the score (same logic as before)
        if response.candidates:
            response_text = response.text.strip()
            score_match = re.search(r'\b([0-3])\b', response_text)
            if score_match:
                score = int(score_match.group(1))
                if 0 <= score <= 3:
                    return score
                else:
                    logging.warning(f"LLM returned score out of range ({score}) for query '{query[:30]}...'. Response: '{response_text}'")
                    return None
            else:
                 logging.warning(f"LLM did not return a valid score (0-3). Query='{query[:30]}...'. Response: '{response_text}'")
                 return None
        else:
            logging.warning(f"LLM response blocked or empty for query '{query[:30]}...'. Prompt feedback: {response.prompt_feedback}")
            return None

    except Exception as e:
        # Catch potential API errors (e.g., invalid model name, permissions)
        logging.error(f"Google Generative AI API call failed for query '{query[:30]}...': {e}")
        return None


# --- Dataset Creation Logic ---
def generate_queries_from_articles(articles: List[Dict], config: Dict) -> List[str]:
    """Generate queries based on article titles and templates."""
    gen_config = config['dataset_generation']
    query_templates = gen_config['query_templates']
    max_queries = gen_config['max_total_queries']
    queries = []
    used_queries = set()

    for article in tqdm(articles, desc="Generating queries"):
        title = article["title"]

        # Add direct title questions if they are questions
        if "?" in title and title not in used_queries:
            queries.append(title)
            used_queries.add(title)

        # Extract potential condition names from title (simple heuristic)
        # TODO: Improve condition extraction (e.g., using NLP/NER)
        words = title.split()
        if len(words) >= 3: # Use titles with enough content
            condition = " ".join(words[:3]).lower().strip('.,:;')
            if condition: # Ensure condition is not empty after stripping
                # Use multiple templates per condition
                for template in query_templates:
                    query = template.format(condition)
                    if query not in used_queries:
                        queries.append(query)
                        used_queries.add(query)
                        if len(queries) >= max_queries:
                            break # Stop if max queries reached
            if len(queries) >= max_queries:
                break # Stop if max queries reached
        if len(queries) >= max_queries:
            break # Stop if max queries reached

    logging.info(f"Generated {len(queries)} initial queries.")
    return queries[:max_queries] # Ensure max_queries limit

def create_relevance_judgments(queries: List[str], documents: Dict[str, str], config: Dict) -> Dict[str, List[str]]:
    """Create relevance judgments using TF-IDF or LLM based on config."""
    gen_config = config['dataset_generation']
    llm_config = config.get('llm', {})
    relevance_llm_config = llm_config.get('relevance', {})
    use_llm = llm_config.get('enabled_relevance', False)

    if not documents:
        logging.warning("No documents provided to create relevance judgments.")
        return {}

    doc_ids = list(documents.keys())
    doc_texts = list(documents.values())
    relevance = {}

    if use_llm:
        logging.info("Generating relevance judgments using LLM (placeholder)...")
        max_pairs = relevance_llm_config.get('max_pairs_per_query', 50)
        # TODO: Implement different sampling strategies based on config
        # sampling_strategy = relevance_llm_config.get('sampling_strategy', 'top_tfidf')

        # Requires TF-IDF calculation even for sampling top candidates
        logging.info("Calculating TF-IDF vectors for LLM candidate sampling...")
        tfidf = TfidfVectorizer(stop_words='english')
        try:
            doc_vectors = tfidf.fit_transform(doc_texts)
        except ValueError as e:
            logging.error(f"TF-IDF Error for LLM sampling: {e}. Cannot generate LLM judgments.")
            return {}

        for query in tqdm(queries, desc="Judging queries via LLM (placeholder)"):
            try:
                query_vector = tfidf.transform([query])
            except ValueError:
                logging.warning(f"Query '{query}' could not be transformed for LLM sampling. Skipping.")
                continue

            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            # Get top N candidates based on TF-IDF to send to LLM
            candidate_indices = np.argsort(similarities)[-max_pairs:][::-1]

            llm_relevant_docs = []
            for idx in candidate_indices:
                doc_id = doc_ids[idx]
                doc_text = documents[doc_id]
                # --- LLM Call Placeholder ---
                llm_score = call_llm_for_relevance(query, doc_text, config)
                # --- End LLM Call Placeholder ---

                # Threshold based on LLM score (e.g., score > 1 means relevant)
                if llm_score is not None and llm_score >= 2: # Example: score 2 or 3 is relevant
                    llm_relevant_docs.append(doc_id)

            if llm_relevant_docs:
                relevance[query] = llm_relevant_docs
            # Optional: Add logging for queries with no LLM-judged relevant docs

    else:
        logging.info("Generating relevance judgments using TF-IDF...")
        tfidf_threshold = gen_config.get('relevance_tfidf_threshold', 0.1)
        top_n = gen_config.get('relevance_top_n', 30)

        tfidf = TfidfVectorizer(stop_words='english')
        try:
            doc_vectors = tfidf.fit_transform(doc_texts)
        except ValueError as e:
            logging.error(f"TF-IDF Error: {e}. Ensure documents are not empty/uniform.")
            return {}

        for query in tqdm(queries, desc="Judging queries via TF-IDF"):
            try:
                query_vector = tfidf.transform([query])
            except ValueError:
                logging.warning(f"Query '{query}' could not be transformed (likely OOV words). Skipping.")
                continue

            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            effective_top_n = min(top_n, len(doc_ids))
            top_indices = np.argsort(similarities)[-effective_top_n:][::-1]

            relevant_docs_for_query = []
            for idx in top_indices:
                if similarities[idx] > tfidf_threshold:
                    relevant_docs_for_query.append(doc_ids[idx])

            if relevant_docs_for_query:
                relevance[query] = relevant_docs_for_query

    logging.info(f"Created relevance judgments for {len(relevance)} queries.")
    return relevance


def create_medical_dataset(config: Dict):
    """Create a medical dataset using PubMed data based on config."""
    api_config = config['pubmed_api']
    gen_config = config['dataset_generation']
    path_config = config['paths']
    topics = gen_config['topics']

    all_articles_details = []
    logging.info(f"Fetching articles for {len(topics)} topics...")
    for topic in tqdm(topics, desc="Processing topics"):
        article_ids = fetch_pubmed_ids(topic, config)
        if article_ids:
            articles = fetch_pubmed_details(article_ids, config)
            if articles:
                all_articles_details.extend(articles)
            time.sleep(api_config['delay_between_topics'])
        else:
             time.sleep(api_config['delay_between_topics']) # Still delay even if no IDs found

    if not all_articles_details:
        logging.error("No articles were fetched. Cannot create dataset. Check API settings and connection.")
        return

    # Remove duplicate articles based on PMID or title/abstract combo
    seen_pmids = set()
    unique_articles = []
    for article in all_articles_details:
        if article['pmid'] not in seen_pmids:
             unique_articles.append(article)
             seen_pmids.add(article['pmid'])
    logging.info(f"Fetched {len(all_articles_details)} articles, {len(unique_articles)} unique.")
    all_articles_details = unique_articles


    # Prepare documents and document IDs
    documents = {f"PMID_{article['pmid']}": f"{article['title']} {article['abstract']}".strip() for article in all_articles_details}
    doc_ids = list(documents.keys())

    # Generate initial queries
    queries = generate_queries_from_articles(all_articles_details, config)

    if not queries:
        logging.error("No queries were generated. Cannot create dataset.")
        return

    # Create relevance judgments
    relevance = create_relevance_judgments(queries, documents, config)

    # Create the final dataset structure
    dataset = {
        "documents": documents, # Store as dict: {doc_id: text}
        "queries": {f"Q_{i+1}": q for i, q in enumerate(queries)}, # Store as dict: {query_id: text}
        "relevance": relevance # Format: {query_text: [doc_id1, doc_id2,...]}
                               # Consider changing key to query_id if needed later
    }

    # Save to file
    output_file = path_config['pubmed_dataset']
    logging.info(f"Saving dataset to {output_file}...")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=4, ensure_ascii=False)
        logging.info(f"Created dataset with {len(dataset['documents'])} documents and {len(dataset['queries'])} queries.")
    except IOError as e:
        logging.error(f"Error writing dataset to file {output_file}: {e}")
    except TypeError as e:
         logging.error(f"Error serializing dataset to JSON: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch PubMed articles and create a medical retrieval dataset.")
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

    # Create the dataset
    create_medical_dataset(config)
    logging.info("Dataset creation process finished.")

# --- (Keep fetch_pubmed_articles function for potential backward compatibility or direct use if needed, but mark as potentially deprecated) ---
# def fetch_pubmed_articles(query: str, max_results: int = 100) -> List[Dict]:
#     """DEPRECATED? Fetch articles from PubMed API. Use fetch_pubmed_ids and fetch_pubmed_details with config instead."""
#     # ... (original implementation remains here) ...
#     pass
# (You can copy the old function body here if you want to keep it accessible, otherwise remove it) 