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

import duckdb
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
import re
from dataclasses import dataclass
from dremioai.config import settings
from dremioai.log import logger
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
from dremioai.memory.embeddings import create_embedding_service, EmbeddingService


@dataclass
class MemoryMatch:
    id: str
    score: float
    text_preview: str
    tags: List[str]
    timestamp: datetime


class MemoryStorage:
    """
    Memory storage using DuckDB with configurable embedding services.
    Supports both TF-IDF and OpenAI embeddings for semantic search.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.config = settings.instance().memory
        if self.config is None:
            # Fallback to default config if not set
            self.config = settings.Memory()
        self.db_path = Path(db_path or self.config.db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize embedding service based on configuration
        self.embedding_service = create_embedding_service(self.config)
        self._vectors_cache = None
        self._texts_cache = None
        self._ids_cache = None

        self._init_db()
    
    def _init_db(self):
        """Initialize the DuckDB database with required tables."""
        conn = duckdb.connect(str(self.db_path))
        try:
            # Create memory table with vector storage
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mem (
                    id TEXT PRIMARY KEY,
                    ts TIMESTAMP DEFAULT now(),
                    text TEXT NOT NULL,
                    tags TEXT[],
                    created_at TIMESTAMP DEFAULT now()
                )
            """)

            # Add vector column if it doesn't exist (migration)
            try:
                conn.execute("ALTER TABLE mem ADD COLUMN vector FLOAT[]")
                logger().info("Added vector column to existing mem table")
            except Exception:
                # Column already exists or other error - that's fine
                pass
            
            # Create metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mem_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            
            # Store configuration metadata
            conn.execute("""
                INSERT OR REPLACE INTO mem_meta (key, value) VALUES 
                ('embedding_model', ?),
                ('embedding_dim', ?),
                ('max_preview_chars', ?)
            """, [
                self.config.embedding_model,
                str(self.config.embedding_dim),
                str(self.config.max_preview_chars)
            ])
            
            conn.commit()
            logger().info(f"Initialized memory database at {self.db_path}")

            # Check if we need to migrate existing memories to have vectors
            self._migrate_existing_memories()
        finally:
            conn.close()

    def _migrate_existing_memories(self):
        """Migrate existing memories that don't have vectors."""
        conn = duckdb.connect(str(self.db_path))
        try:
            # Find memories without vectors
            result = conn.execute("""
                SELECT id, text FROM mem WHERE vector IS NULL
            """).fetchall()

            if not result:
                return

            logger().info(f"Migrating {len(result)} memories to add vectors...")

            # First, fit the embedding service on all existing texts
            all_texts = [text for _, text in result]
            if all_texts:
                self.embedding_service.fit(all_texts)

            # Generate and store vectors for each memory
            for memory_id, text in result:
                try:
                    vector = self.embedding_service.embed_text(text)
                    vector_list = vector.tolist() if hasattr(vector, 'tolist') else vector.flatten().tolist()

                    conn.execute("""
                        UPDATE mem SET vector = ? WHERE id = ?
                    """, [vector_list, memory_id])

                except Exception as e:
                    logger().error(f"Failed to generate vector for memory {memory_id}: {e}")

            conn.commit()
            logger().info(f"Successfully migrated {len(result)} memories with vectors")

        except Exception as e:
            logger().error(f"Error during memory migration: {e}")
        finally:
            conn.close()

    def force_remigration(self):
        """Force re-migration of all memories (useful when changing embedding types)."""
        conn = duckdb.connect(str(self.db_path))
        try:
            # Clear all existing vectors
            conn.execute("UPDATE mem SET vector = NULL")
            conn.commit()
            logger().info("Cleared all existing vectors")

            # Trigger migration
            self._migrate_existing_memories()

        except Exception as e:
            logger().error(f"Error during forced remigration: {e}")
        finally:
            conn.close()

    def _rebuild_vector_cache(self):
        """Rebuild the TF-IDF vector cache from all stored memories."""
        conn = duckdb.connect(str(self.db_path))
        try:
            result = conn.execute("SELECT id, text FROM mem ORDER BY created_at").fetchall()

            if not result:
                self._vectors_cache = None
                self._texts_cache = []
                self._ids_cache = []
                return

            # Extract texts and IDs
            ids, texts = zip(*result)
            self._ids_cache = list(ids)
            self._texts_cache = list(texts)

            # Fit embedding service and generate embeddings
            self.embedding_service.fit(texts)
            embeddings = self.embedding_service.embed_texts(texts)
            self._vectors_cache = embeddings

            logger().info(f"Rebuilt vector cache with {len(texts)} memories using {type(self.embedding_service).__name__}")

        finally:
            conn.close()

    def _ensure_vector_cache(self):
        """Ensure the vector cache is built and up-to-date."""
        if self._vectors_cache is None:
            self._rebuild_vector_cache()

    def _generate_vector(self, text: str) -> np.ndarray:
        """Generate a vector for a single text using the embedding service."""
        # Ensure we have a fitted embedding service
        self._ensure_vector_cache()

        # Generate embedding for the text
        return self.embedding_service.embed_text(text)
    
    def put_memory(self, text: str, id: Optional[str] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Store a memory with optional ID and tags.
        
        Args:
            text: The text content to store
            id: Optional ID, will generate UUID if not provided
            tags: Optional list of tags
            
        Returns:
            Dictionary with id, tokens_est, embedding_dim, stored status
        """
        if id is None:
            id = str(uuid.uuid4())
        
        if tags is None:
            tags = []
        
        # Generate vector for the text
        vector = self._generate_vector(text)
        vector_list = vector.tolist() if hasattr(vector, 'tolist') else vector.flatten().tolist()

        conn = duckdb.connect(str(self.db_path))
        try:
            # Insert or replace the memory with vector
            conn.execute("""
                INSERT OR REPLACE INTO mem (id, text, tags, vector, ts, created_at)
                VALUES (?, ?, ?, ?, now(), now())
            """, [id, text, tags, vector_list])

            conn.commit()

            # Invalidate vector cache since we added new data
            self._vectors_cache = None
            self._texts_cache = None
            self._ids_cache = None

            # Estimate tokens (rough approximation: ~4 chars per token)
            tokens_est = len(text) // 4

            logger().info(f"Stored memory with id={id}, length={len(text)}, vector_dim={len(vector_list)}")

            return {
                "id": id,
                "tokens_est": tokens_est,
                "embedding_dim": self.embedding_service.get_dimension(),
                "stored": True
            }
        finally:
            conn.close()
    
    def _calculate_vector_similarity(self, query: str, tag_filter: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Calculate TF-IDF vector similarity between query and all stored memories.
        Returns list of (memory_id, similarity_score) tuples sorted by similarity.
        """
        # Generate query vector
        query_vector = self._generate_vector(query)

        # Get all memories with vectors from database
        conn = duckdb.connect(str(self.db_path))
        try:
            # Build SQL query with optional tag filtering
            sql = "SELECT id, vector FROM mem WHERE vector IS NOT NULL"
            params = []

            if tag_filter:
                # Filter by tags (any tag in the filter list)
                tag_conditions = []
                for tag in tag_filter:
                    tag_conditions.append("list_contains(tags, ?)")
                    params.append(tag)
                sql += f" AND ({' OR '.join(tag_conditions)})"

            result = conn.execute(sql, params).fetchall()

            if not result:
                return []

            # Calculate similarities
            scored_results = []
            for memory_id, stored_vector in result:
                if stored_vector is None:
                    continue

                # Convert stored vector to numpy array
                stored_vector = np.array(stored_vector)

                # Calculate cosine similarity using embedding service
                similarity = self.embedding_service.calculate_similarity(query_vector, stored_vector.reshape(1, -1))[0]
                scored_results.append((memory_id, float(similarity)))

            # Sort by similarity descending
            scored_results.sort(key=lambda x: x[1], reverse=True)
            return scored_results

        except Exception as e:
            logger().error(f"Error calculating vector similarity: {e}")
            return []
        finally:
            conn.close()
    
    def search_memory(self, query: str, top_k: int = 5, tag_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search for memories using TF-IDF vector similarity.

        Args:
            query: Search query string
            top_k: Number of top results to return
            tag_filter: Optional list of tags to filter by

        Returns:
            Dictionary with matches and query info
        """
        # Ensure top_k is an integer
        top_k = int(top_k) if top_k is not None else 5

        # Get vector similarity scores (tag filtering applied inside)
        similarity_results = self._calculate_vector_similarity(query, tag_filter)

        if not similarity_results:
            return {
                "matches": [],
                "query_embedding_dim": self.config.embedding_dim
            }

        # Get memory details from database
        conn = duckdb.connect(str(self.db_path))
        try:
            # Get all memory details we need
            memory_ids = [memory_id for memory_id, _ in similarity_results[:top_k]]
            if not memory_ids:
                return {
                    "matches": [],
                    "query_embedding_dim": self.config.embedding_dim
                }

            # Build query to get memory details
            placeholders = ','.join(['?' for _ in memory_ids])
            result = conn.execute(f"""
                SELECT id, text, tags, ts FROM mem
                WHERE id IN ({placeholders})
            """, memory_ids).fetchall()

            # Create lookup for database results
            db_lookup = {}
            for row in result:
                id, text, tags, ts = row
                db_lookup[id] = {
                    'text': text,
                    'tags': tags or [],
                    'ts': ts
                }

            # Process similarity results
            matches = []
            for memory_id, score in similarity_results[:top_k]:
                # Skip very low similarity scores
                if score < 0.01:
                    continue

                if memory_id not in db_lookup:
                    continue

                memory_data = db_lookup[memory_id]

                # Create preview text
                max_chars = self.config.max_preview_chars
                if not isinstance(max_chars, int):
                    max_chars = int(max_chars) if max_chars else 240
                preview = memory_data['text'][:max_chars]
                if len(memory_data['text']) > max_chars:
                    preview += "..."

                matches.append({
                    "id": memory_id,
                    "score": score,
                    "text_preview": preview,
                    "tags": memory_data['tags']
                })

            logger().info(f"Found {len(matches)} matches for query: {query} (vector similarity)")

            return {
                "matches": matches,
                "query_embedding_dim": self.embedding_service.get_dimension()
            }
        finally:
            conn.close()
    
    def get_memory(self, id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory by ID.
        
        Args:
            id: Memory ID
            
        Returns:
            Dictionary with memory details or None if not found
        """
        conn = duckdb.connect(str(self.db_path))
        try:
            result = conn.execute("""
                SELECT id, text, tags, ts FROM mem WHERE id = ?
            """, [id]).fetchone()
            
            if result:
                id, text, tags, ts = result
                return {
                    "id": id,
                    "text": text,
                    "tags": tags or [],
                    "ts": ts.isoformat() if ts else None
                }
            return None
        finally:
            conn.close()
    
    def list_memories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all memories with basic info."""
        conn = duckdb.connect(str(self.db_path))
        try:
            result = conn.execute("""
                SELECT id, text, tags, ts FROM mem 
                ORDER BY ts DESC 
                LIMIT ?
            """, [limit]).fetchall()
            
            memories = []
            for row in result:
                id, text, tags, ts = row
                max_chars = self.config.max_preview_chars
                if not isinstance(max_chars, int):
                    max_chars = int(max_chars) if max_chars else 240
                preview = text[:max_chars]
                if len(text) > max_chars:
                    preview += "..."
                
                memories.append({
                    "id": id,
                    "text_preview": preview,
                    "tags": tags or [],
                    "ts": ts.isoformat() if ts else None
                })
            
            return memories
        finally:
            conn.close()
