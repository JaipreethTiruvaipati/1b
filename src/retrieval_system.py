import os
import numpy as np
# Try to import faiss, but have a fallback
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
import time
from typing import List, Dict, Any, Union, Tuple
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Try to import sentence_transformers, but have a fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class SimpleSearchIndex:
    """
    Simple search index using cosine similarity when FAISS is not available.
    """
    
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        self.embeddings = None
        self.doc_ids = []
    
    def add_with_ids(self, embeddings, ids):
        """Add embeddings with IDs."""
        if self.embeddings is None:
            self.embeddings = embeddings
            self.doc_ids = ids.tolist()
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
            self.doc_ids.extend(ids.tolist())
    
    def search(self, query_embedding, k):
        """Search for similar embeddings."""
        if self.embeddings is None:
            return np.array([[0.0]]), np.array([[0]])
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        # Convert similarities to distances (1 - similarity)
        distances = 1 - similarities[top_indices]
        
        return distances.reshape(1, -1), np.array([self.doc_ids[i] for i in top_indices]).reshape(1, -1)


class SimpleTextEmbedder:
    """
    Simple text embedding system using TF-IDF when SentenceTransformers is not available.
    """
    
    def __init__(self, max_features=5000):
        """Initialize the simple embedder."""
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.is_fitted = False
        self.embedding_dim = max_features
    
    def fit(self, texts: List[str]):
        """Fit the vectorizer on the provided texts."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
    
    def encode(self, texts: Union[str, List[str]], convert_to_numpy=True):
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        if not self.is_fitted:
            self.fit(texts)
        
        # Transform to TF-IDF vectors
        embeddings = self.vectorizer.transform(texts)
        
        if convert_to_numpy:
            return embeddings.toarray().astype(np.float32)
        return embeddings
    
    def get_sentence_embedding_dimension(self):
        """Get the embedding dimension."""
        return self.embedding_dim


class Retriever:
    """
    Document retrieval system using SentenceTransformers and FAISS.
    This class handles embedding generation and approximate nearest neighbor search.
    All model loading is now offline, local, and CPU-only.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the retriever with a SentenceTransformer model or fallback.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Try to use SentenceTransformers first
        self.model = None
        self.use_offline_mode = False
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Set transformers to offline mode
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_HUB_OFFLINE"] = "1"
                
                # Use local model path if available
                local_model_path = self.model_dir / model_name.replace('/', '_')
                if local_model_path.exists():
                    print(f"Loading local model from {local_model_path}")
                    self.model = SentenceTransformer(str(local_model_path), local_files_only=True)
                else:
                    print(f"Attempting to load model {model_name} in offline mode")
                    self.model = SentenceTransformer(model_name, local_files_only=True)
                
                if hasattr(self.model, 'to'):
                    self.model = self.model.to('cpu')
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                print("Successfully loaded SentenceTransformers model")
                
            except Exception as e:
                print(f"Failed to load SentenceTransformers model: {e}")
                print("Falling back to simple TF-IDF embeddings")
                self.model = None
                self.use_offline_mode = True
        else:
            print("SentenceTransformers not available, using TF-IDF fallback")
            self.use_offline_mode = True
        
        # Use fallback if SentenceTransformers failed
        if self.model is None or self.use_offline_mode:
            self.model = SimpleTextEmbedder()
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
    
    def _convert_to_onnx(self):
        """
        Convert the model to ONNX format for faster inference.
        """
        # TODO: Implement ONNX conversion logic
        pass
    
    def index_documents(self, documents: List[str]):
        """
        Index a list of documents for later retrieval.
        
        Args:
            documents: List of document texts to index
        """
        # Store documents
        self.documents = documents
        
        # Generate embeddings
        embeddings = self.generate_embeddings(documents)
        
        # Create and populate index
        if FAISS_AVAILABLE:
            # Use FAISS index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index = faiss.IndexIDMap(self.index)
            
            for i, embedding in enumerate(embeddings):
                self.index.add_with_ids(np.array([embedding]), np.array([i]))
        else:
            # Use simple fallback index
            print("FAISS not available, using simple cosine similarity search")
            self.index = SimpleSearchIndex(self.embedding_dim)
            
            for i, embedding in enumerate(embeddings):
                self.index.add_with_ids(np.array([embedding]), np.array([i]))
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            NumPy array of embeddings
        """
        if isinstance(self.model, SimpleTextEmbedder):
            return self.model.encode(texts, convert_to_numpy=True)
        else:
            # SentenceTransformers model
            try:
                return self.model.encode(texts, show_progress_bar=True)
            except:
                return self.model.encode(texts)
    
    def search(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for similar documents to a query.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of (document_id, distance) tuples
        """
        if self.index is None:
            raise ValueError("No documents indexed yet")
        
        # Generate query embedding
        query_embedding = self.generate_embeddings([query])
        
        # Search for similar documents
        distances, indices = self.index.search(query_embedding, k)
        
        # Return as list of (document_id, distance) tuples
        return [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0])]


class SimpleReranker:
    """
    Simple reranker using TF-IDF similarity when sophisticated models aren't available.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    def rerank(self, query: str, documents: List[str], scores: List[float] = None) -> List[Tuple[int, float]]:
        """Simple reranking using TF-IDF similarity."""
        if not documents:
            return []
        
        # Fit vectorizer on all texts
        all_texts = [query] + documents
        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Get query vector (first row)
            query_vec = tfidf_matrix[0:1]
            
            # Get document vectors
            doc_vecs = tfidf_matrix[1:]
            
            # Compute similarities
            similarities = cosine_similarity(query_vec, doc_vecs)[0]
            
            # Create (document_id, score) tuples
            results = [(i, float(sim)) for i, sim in enumerate(similarities)]
            
            # Sort by score in descending order
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results
        except:
            # Fallback: return original order with dummy scores
            return [(i, 0.5) for i in range(len(documents))]


class Reranker:
    """
    Document reranking system using cross-encoders.
    This class handles reranking of initial retrieval results.
    All model loading is now offline, local, and CPU-only.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        """
        Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        self.model = None
        self.use_simple_reranker = False
        
        # Try to load sophisticated models first
        if TRANSFORMERS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                os.environ["HF_HUB_OFFLINE"] = "1"
                
                local_model_path = self.model_dir / model_name.replace('/', '_')
                
                if local_model_path.exists():
                    if TRANSFORMERS_AVAILABLE:
                        self.tokenizer = AutoTokenizer.from_pretrained(str(local_model_path), local_files_only=True)
                    try:
                        from sentence_transformers.cross_encoder import CrossEncoder
                        self.model = CrossEncoder(str(local_model_path))
                    except ImportError:
                        print("CrossEncoder not available, falling back to simple reranker")
                        self.use_simple_reranker = True
                else:
                    print("Local model not found, using simple reranker")
                    self.use_simple_reranker = True
                    
                if self.model is not None and hasattr(self.model, 'to'):
                    self.model = self.model.to('cpu')
                    
            except Exception as e:
                print(f"Failed to load reranker model: {e}")
                print("Using simple reranker")
                self.use_simple_reranker = True
        else:
            print("Required libraries not available, using simple reranker")
            self.use_simple_reranker = True
        
        # Use simple reranker as fallback
        if self.use_simple_reranker or self.model is None:
            self.simple_reranker = SimpleReranker()
    
    def _convert_to_onnx(self):
        """
        Convert the model to ONNX format for faster inference.
        """
        # TODO: Implement ONNX conversion logic
        pass
    
    def rerank(self, query: str, documents: List[str], scores: List[float] = None) -> List[Tuple[int, float]]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Query string
            documents: List of document texts to rerank
            scores: Initial retrieval scores (optional)
            
        Returns:
            List of (document_id, score) tuples
        """
        # Use simple reranker if sophisticated model not available
        if self.use_simple_reranker or self.model is None:
            return self.simple_reranker.rerank(query, documents, scores)
        
        # Prepare inputs for cross-encoder
        inputs = [(query, doc) for doc in documents]
        
        # Score the inputs using the model
        try:
            scores = self.model.predict(inputs)
        except Exception as e:
            print(f"Error during reranking: {e}, using simple reranker")
            return self.simple_reranker.rerank(query, documents, scores)
        
        # Create (document_id, score) tuples
        results = [(i, float(score)) for i, score in enumerate(scores)]
        
        # Sort by score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results


class RetrievalSystem:
    """
    Complete retrieval system combining the retriever and reranker.
    All model loading is now offline, local, and CPU-only.
    """
    
    def __init__(
        self, 
        retriever_model: str = "sentence-transformers/all-MiniLM-L6-v2", 
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    ):
        """
        Initialize the retrieval system.
        
        Args:
            retriever_model: Name of the SentenceTransformer model to use
            reranker_model: Name of the cross-encoder model to use
        """
        self.retriever = Retriever(retriever_model)
        self.reranker = Reranker(reranker_model)
        self.document_texts = []
        self.document_metadata = []
    
    def index_sections(self, sections: List[Any]):
        """
        Index sections for retrieval.
        
        Args:
            sections: List of Section objects
        """
        # Extract texts and metadata
        texts = []
        metadata = []
        
        for i, section in enumerate(sections):
            section_text = section.section_title + "\n" + section.text
            texts.append(section_text)
            metadata.append({
                "index": i,
                "document": section.document,
                "section_title": section.section_title,
                "page_number": section.page_number
            })
        
        # Store document texts and metadata
        self.document_texts = texts
        self.document_metadata = metadata
        
        # Index documents in retriever
        self.retriever.index_documents(texts)
    
    def search(self, query: str, k: int = 10, rerank: bool = True) -> List[Dict[str, Any]]:
        """
        Search for relevant sections.
        
        Args:
            query: Query string
            k: Number of results to return
            rerank: Whether to rerank results
            
        Returns:
            List of search results with metadata
        """
        # Retrieve initial results
        retrieval_results = self.retriever.search(query, k=k)
        
        # Get document texts for retrieved results
        retrieved_ids = [idx for idx, _ in retrieval_results]
        retrieved_docs = [self.document_texts[idx] for idx in retrieved_ids]
        
        if rerank and retrieved_docs:
            # Rerank results
            rerank_results = self.reranker.rerank(query, retrieved_docs)
            
            # Map back to original document IDs
            reranked_ids = [retrieved_ids[idx] for idx, _ in rerank_results]
            reranked_scores = [score for _, score in rerank_results]
            
            # Prepare results
            results = []
            for idx, score in zip(reranked_ids, reranked_scores):
                result = self.document_metadata[idx].copy()
                result["score"] = score
                results.append(result)
        else:
            # Use retrieval results directly
            results = []
            for idx, score in retrieval_results:
                result = self.document_metadata[idx].copy()
                result["score"] = -score  # Convert distance to similarity score
                results.append(result)
        
        return results 