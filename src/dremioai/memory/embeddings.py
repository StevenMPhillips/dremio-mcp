#
#  Copyright (C) 2017-2025 Dremio Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
from dremioai.config import settings
from dremioai.log import logger


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    @abstractmethod
    def fit(self, texts: List[str]) -> None:
        """Fit the embedding model on a corpus of texts."""
        pass
    
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        pass
    
    def calculate_similarity(self, query_embedding: np.ndarray, stored_embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between query and stored embeddings."""
        return cosine_similarity([query_embedding], stored_embeddings).flatten()


class TFIDFEmbeddingService(EmbeddingService):
    """TF-IDF based embedding service using scikit-learn."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents='ascii'
        )
        self._fitted = False
    
    def fit(self, texts: List[str]) -> None:
        """Fit the TF-IDF vectorizer on the corpus."""
        if texts:
            self.vectorizer.fit(texts)
            self._fitted = True
            logger().info(f"Fitted TF-IDF vectorizer on {len(texts)} texts")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate TF-IDF embedding for a single text."""
        if not self._fitted:
            # If not fitted, fit on this single text
            self.fit([text])
        
        vector = self.vectorizer.transform([text])
        return vector.toarray().flatten()
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate TF-IDF embeddings for multiple texts."""
        if not self._fitted:
            self.fit(texts)
        
        vectors = self.vectorizer.transform(texts)
        return vectors.toarray()
    
    def get_dimension(self) -> int:
        """Get the dimension of TF-IDF vectors."""
        if self._fitted:
            return len(self.vectorizer.get_feature_names_out())
        return 1000  # Default max_features


class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI embedding service using text-embedding-3-small."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self._dimension = None
        logger().info(f"Initialized OpenAI embedding service with model: {model}")
    
    def fit(self, texts: List[str]) -> None:
        """OpenAI embeddings don't require fitting, but we can determine dimension."""
        if texts and self._dimension is None:
            # Get dimension from first embedding
            sample_embedding = self.embed_text(texts[0])
            self._dimension = len(sample_embedding)
            logger().info(f"Determined OpenAI embedding dimension: {self._dimension}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate OpenAI embedding for a single text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float"
            )
            embedding = np.array(response.data[0].embedding)
            
            if self._dimension is None:
                self._dimension = len(embedding)
            
            return embedding
            
        except Exception as e:
            logger().error(f"Failed to generate OpenAI embedding: {e}")
            raise
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate OpenAI embeddings for multiple texts."""
        try:
            # OpenAI API supports batch requests
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            
            embeddings = []
            for data in response.data:
                embeddings.append(data.embedding)
            
            embeddings_array = np.array(embeddings)
            
            if self._dimension is None:
                self._dimension = embeddings_array.shape[1]
            
            logger().info(f"Generated {len(embeddings)} OpenAI embeddings")
            return embeddings_array
            
        except Exception as e:
            logger().error(f"Failed to generate OpenAI embeddings: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get the dimension of OpenAI embeddings."""
        if self._dimension is None:
            # text-embedding-3-small has 1536 dimensions
            return 1536
        return self._dimension


def create_embedding_service(config: Optional[object] = None) -> EmbeddingService:
    """Factory function to create the appropriate embedding service."""
    if config is None:
        config = settings.instance().memory
    
    embedding_type = getattr(config, 'embedding_type', 'tfidf')
    
    if embedding_type == 'openai':
        api_key = getattr(config, 'openai_api_key', None)
        if not api_key:
            logger().warning("OpenAI API key not configured, falling back to TF-IDF")
            return TFIDFEmbeddingService()
        
        model = getattr(config, 'openai_model', 'text-embedding-3-small')
        return OpenAIEmbeddingService(api_key=api_key, model=model)
    
    else:  # Default to TF-IDF
        return TFIDFEmbeddingService()
