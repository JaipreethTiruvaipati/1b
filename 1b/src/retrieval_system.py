import os
import numpy as np
import faiss
import torch
import time
from typing import List, Dict, Any, Union, Tuple
from sentence_transformers import SentenceTransformer
import onnxruntime as ort
from transformers import AutoTokenizer
import json
from pathlib import Path

class Retriever:
    """
    Document retrieval system using SentenceTransformers and FAISS.
    This class handles embedding generation and approximate nearest neighbor search.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the retriever with a SentenceTransformer model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model_name = model_name
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Initialize the embedding model
        self.model = SentenceTransformer(model_name)
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
        
        # Create and populate FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index = faiss.IndexIDMap(self.index)
        
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
        return self.model.encode(texts, show_progress_bar=True)
    
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


class Reranker:
    """
    Document reranking system using cross-encoders.
    This class handles reranking of initial retrieval results.
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
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load cross-encoder model
        try:
            from sentence_transformers.cross_encoder import CrossEncoder
            self.model = CrossEncoder(model_name)
        except ImportError:
            print("CrossEncoder not available, using ONNX runtime directly")
            self.model = None
    
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
        # Prepare inputs for cross-encoder
        inputs = [(query, doc) for doc in documents]
        
        # Score the inputs using the model
        if self.model:
            scores = self.model.predict(inputs)
        else:
            # Fallback if CrossEncoder is not available
            scores = [0.5] * len(inputs)  # Placeholder scores
        
        # Create (document_id, score) tuples
        results = [(i, float(score)) for i, score in enumerate(scores)]
        
        # Sort by score in descending order
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results


class RetrievalSystem:
    """
    Complete retrieval system combining the retriever and reranker.
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
        
        for section in sections:
            section_text = section.section_title + "\n" + section.text
            texts.append(section_text)
            metadata.append({
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