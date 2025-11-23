"""Gemini API-based embedding service to replace local Sentence Transformers.

This eliminates the 80-120MB memory footprint of the local model
and uses Google's Gemini text-embedding-004 API instead.
"""

import os
from typing import List, Optional
import numpy as np
import google.generativeai as genai


class GeminiEmbeddingService:
    """Embedding service using Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini embedding service.
        
        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter. Get free key at: https://aistudio.google.com/app/apikey"
            )
        
        genai.configure(api_key=self.api_key)
        self.model_name = "models/text-embedding-004"
        print(f"✓ Gemini Embeddings initialized (model: {self.model_name})")
    
    def embed_texts(self, texts: list[str], task_type: str = "retrieval_document") -> np.ndarray:
        """Embed multiple texts using Gemini API.
        
        Args:
            texts: List of texts to embed
            task_type: 'retrieval_document' for passages, 'retrieval_query' for questions
            
        Returns:
            NumPy array of embeddings (shape: [len(texts), 768])
        """
        embeddings = []
        
        # Smaller batch size to reduce memory usage on free tier
        batch_size = 20  # Reduced from 100 to 20
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Process even smaller batches to minimize memory spikes
            for text in batch:
                try:
                    result = genai.embed_content(
                        model=self.model_name,
                        content=text,
                        task_type=task_type
                    )
                    embeddings.append(result['embedding'])
                except Exception as e:
                    print(f"Warning: Failed to embed text (using zeros): {e}")
                    # Return zero vector on failure (768 dims for text-embedding-004)
                    embeddings.append([0.0] * 768)
            
            # More frequent progress updates and memory cleanup
            if i + batch_size < len(texts):
                print(f"  Embedded {i + batch_size}/{len(texts)} texts...")
                # Force garbage collection to free memory
                import gc
                gc.collect()
        
        # Convert to numpy array and normalize
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Normalize embeddings (unit vectors for cosine similarity)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings_array = embeddings_array / norms
        
        return embeddings_array
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query using Gemini API.
        
        Args:
            query: Question text
            
        Returns:
            NumPy array embedding (shape: [768])
        """
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=query,
                task_type="retrieval_query"
            )
            embedding = np.array(result['embedding'], dtype=np.float32)
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
        except Exception as e:
            print(f"Error embedding query: {e}")
            # Return zero vector on failure
            return np.zeros(768, dtype=np.float32)


# Type hint import
from typing import Optional
