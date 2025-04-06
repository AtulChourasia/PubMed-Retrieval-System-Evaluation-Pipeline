from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from transformers import AutoTokenizer, AutoModel # Unused
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer # Unused
import logging # Added for potential logging needs

# --- Base Class ---
class BaseRetriever:
    """Base class for all retriever models."""
    def __init__(self):
        self.documents: List[str] = []
        self.doc_ids: List[str] = []
        # self.embeddings = None # Moved to DenseRetriever

    def index(self, documents: List[str], doc_ids: List[str]):
        """Indexes the provided documents and their IDs."""
        if len(documents) != len(doc_ids):
            raise ValueError("Number of documents must match number of document IDs.")
        self.documents = documents
        self.doc_ids = doc_ids

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Retrieves the top k document IDs and scores for a given query."""
        raise NotImplementedError("Subclasses must implement the retrieve method.")

# --- BM25 Retriever ---
class BM25Retriever(BaseRetriever):
    """Retriever using the BM25Okapi algorithm."""
    def __init__(self):
        super().__init__()
        self.bm25 = None

    def index(self, documents: List[str], doc_ids: List[str]):
        """Tokenizes and indexes documents for BM25."""
        super().index(documents, doc_ids)
        logging.info("Indexing documents for BM25...")
        # Consider more sophisticated tokenization if needed (e.g., removing punctuation)
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        logging.info("BM25 indexing complete.")

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Retrieves documents using BM25 scores."""
        if not self.bm25:
            raise RuntimeError("BM25 model has not been indexed. Call index() first.")
        tokenized_query = query.lower().split()
        try:
            doc_scores = self.bm25.get_scores(tokenized_query)
            # Get top k indices, handling cases where k > number of docs
            effective_k = min(k, len(self.doc_ids))
            top_k_indices = np.argsort(doc_scores)[-effective_k:][::-1]
            return [(self.doc_ids[i], doc_scores[i]) for i in top_k_indices]
        except Exception as e:
            logging.error(f"Error during BM25 retrieval for query '{query}': {e}")
            return []

# --- Dense Retriever ---
class DenseRetriever(BaseRetriever):
    """Retriever using Sentence Transformers for dense embeddings."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.embeddings: np.ndarray | None = None # Initialize embeddings here
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading Sentence Transformer model {model_name} onto {self.device}")
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            logging.error(f"Failed to load Sentence Transformer model {model_name}: {e}")
            raise
        logging.info(f"Model {model_name} loaded successfully.")

    def index(self, documents: List[str], doc_ids: List[str]):
        """Encodes documents into dense vectors."""
        super().index(documents, doc_ids)
        logging.info(f"Encoding {len(documents)} documents using {self.model_name}...")
        self.embeddings = self.model.encode(documents, show_progress_bar=True, convert_to_numpy=True)
        logging.info("Document encoding complete.")

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Retrieves documents based on cosine similarity of embeddings."""
        if self.embeddings is None:
            raise RuntimeError("Dense model has not been indexed. Call index() first.")
        try:
            query_embedding = self.model.encode(query, convert_to_numpy=True).reshape(1, -1)
            # Using cosine_similarity for clarity, though dot product is equivalent for normalized embeddings
            scores = cosine_similarity(query_embedding, self.embeddings).flatten()
            # Get top k indices, handling cases where k > number of docs
            effective_k = min(k, len(self.doc_ids))
            top_k_indices = np.argsort(scores)[-effective_k:][::-1]
            return [(self.doc_ids[i], scores[i]) for i in top_k_indices]
        except Exception as e:
            logging.error(f"Error during Dense retrieval for query '{query}': {e}")
            return []

# --- Domain-Specific Dense Retrievers ---
class MedicalDenseRetriever(DenseRetriever):
    """Dense retriever using a PubMedBERT model."""
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"):
        super().__init__(model_name)

class ClinicalDenseRetriever(DenseRetriever):
    """Dense retriever using a ClinicalBERT model."""
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT"):
        super().__init__(model_name)

class BioBERTDenseRetriever(DenseRetriever):
    """Dense retriever using a BioBERT model."""
    def __init__(self, model_name: str = "dmis-lab/biobert-base-cased-v1.2"):
        super().__init__(model_name)

class PubMedBERTDenseRetriever(DenseRetriever):
    """Dense retriever using a different PubMedBERT variant."""
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        super().__init__(model_name)

# --- Hybrid Retriever ---
def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize scores between 0 and 1."""
    if not scores:
        return {}
    values = np.array(list(scores.values()))
    min_val = values.min()
    max_val = values.max()
    if max_val == min_val:
        # Handle case where all scores are the same (avoid division by zero)
        # Assign 0.5 or keep original score? Assigning 0.5 seems reasonable.
        return {doc_id: 0.5 for doc_id in scores}
    normalized_values = (values - min_val) / (max_val - min_val)
    return dict(zip(scores.keys(), normalized_values))

class HybridRetriever(BaseRetriever):
    """Combines BM25 and Dense retrieval scores."""
    def __init__(self, alpha: float = 0.5, dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("Alpha must be between 0 and 1 for hybrid retrieval.")
        self.alpha = alpha # Weight for BM25
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(model_name=dense_model_name)

    def index(self, documents: List[str], doc_ids: List[str]):
        """Indexes documents using both BM25 and Dense models."""
        super().index(documents, doc_ids)
        logging.info("Indexing for Hybrid Retriever...")
        self.bm25.index(documents, doc_ids)
        self.dense.index(documents, doc_ids)
        logging.info("Hybrid indexing complete.")

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Retrieves documents by combining normalized BM25 and Dense scores."""
        # Retrieve a larger candidate pool (e.g., 2*k or a fixed number like 100)
        # This provides a better basis for score normalization.
        candidate_pool_size = max(k * 2, 50) # Retrieve more candidates

        bm25_results = dict(self.bm25.retrieve(query, candidate_pool_size))
        dense_results = dict(self.dense.retrieve(query, candidate_pool_size))

        # Normalize scores separately
        norm_bm25_scores = _normalize_scores(bm25_results)
        norm_dense_scores = _normalize_scores(dense_results)

        # Combine scores
        combined_scores = {}
        all_docs = set(norm_bm25_scores.keys()) | set(norm_dense_scores.keys())

        for doc_id in all_docs:
            bm25_score = norm_bm25_scores.get(doc_id, 0) # Default to 0 if not found
            dense_score = norm_dense_scores.get(doc_id, 0) # Default to 0 if not found
            combined_scores[doc_id] = self.alpha * bm25_score + (1 - self.alpha) * dense_score

        # Sort by combined score and return top k
        sorted_docs = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_docs[:k]

# --- Placeholder for Re-Ranker (Requires User Implementation) ---
class CrossEncoderReRanker:
    """Placeholder for a cross-encoder re-ranking model."""
    def __init__(self, model_name: str, device: torch.device | str = 'cpu'):
        """Initializes the re-ranker.

        Args:
            model_name: The name of the cross-encoder model (e.g., from Hugging Face).
            device: The device to run the model on ('cpu', 'cuda').
        """
        self.model_name = model_name
        self.device = device
        # !!! USER IMPLEMENTATION NEEDED HERE !!!
        # 1. Load the cross-encoder model using Hugging Face's CrossEncoder
        #    or AutoModelForSequenceClassification + AutoTokenizer
        #    (Requires sentence-transformers or transformers library)
        # Example (using sentence_transformers):
        # from sentence_transformers.cross_encoder import CrossEncoder
        # self.model = CrossEncoder(model_name, device=self.device, max_length=512) # Adjust max_length
        logging.info(f"Placeholder: Initialized CrossEncoderReRanker with model {model_name} on device {device}")
        self.model = None # Placeholder

    def rerank(self, query: str, documents: Dict[str, str], top_k: int = 50) -> List[Tuple[str, float]]:
        """Re-ranks a list of documents for a given query.

        Args:
            query: The query text.
            documents: A dictionary {doc_id: doc_text} of candidate documents.
            top_k: The number of documents to return after re-ranking.

        Returns:
            A list of (doc_id, score) tuples, sorted by relevance score.
        """
        if not self.model:
            logging.error("Re-ranker model not loaded. Cannot rerank.")
            # Fallback: return original docs maybe? Or empty?
            return sorted(documents.items(), key=lambda x: 0.5, reverse=True)[:top_k] # Random sort placeholder

        if not documents:
            return []

        doc_ids = list(documents.keys())
        doc_texts = list(documents.values())

        # Prepare pairs for the cross-encoder
        sentence_pairs = [[query, doc_text] for doc_text in doc_texts]

        # !!! USER IMPLEMENTATION NEEDED HERE !!!
        # 1. Predict scores using the loaded cross-encoder model
        # Example (using sentence_transformers):
        # scores = self.model.predict(sentence_pairs, show_progress_bar=False)

        # Placeholder scores (e.g., random or based on length)
        logging.warning("CrossEncoderReRanker.rerank not implemented. Using placeholder scores.")
        scores = np.random.rand(len(doc_ids))

        # Combine doc_ids with scores and sort
        scored_docs = list(zip(doc_ids, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:top_k] 