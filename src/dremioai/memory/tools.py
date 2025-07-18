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

from typing import Dict, Any, List, Optional, ClassVar, Annotated
from dremioai.tools.tools import Tools
from dremioai.config.tools import ToolType
from dremioai.memory.storage import MemoryStorage
from dremioai.log import logger


class PutMemory(Tools):
    """Store an enterprise memory (metric definition, glossary note) for future retrieval."""
    
    For: ClassVar[Annotated[ToolType, ToolType.FOR_SELF | ToolType.FOR_DATA_PATTERNS]]
    
    def __init__(self):
        super().__init__()
        self._storage = None
    
    @property
    def storage(self) -> MemoryStorage:
        """Lazy initialization of storage."""
        if self._storage is None:
            self._storage = MemoryStorage()
        return self._storage
    
    async def invoke(
        self, 
        text: str, 
        id: Optional[str] = None, 
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Store an enterprise memory (metric definition, glossary note) so it can be used in future answers.
        
        This tool allows you to persist important business definitions, metric calculations,
        data glossaries, and other reusable knowledge that should be remembered across sessions.
        
        Args:
            text: The text content to store (metric definition, note, etc.)
            id: Optional unique identifier. If not provided, a UUID will be generated
            tags: Optional list of tags to categorize the memory (e.g., ["metric", "finance", "ARR"])
            
        Returns:
            Dictionary containing:
            - id: The unique identifier for the stored memory
            - tokens_est: Estimated token count of the stored text
            - embedding_dim: Dimension of the embedding space used
            - stored: Boolean indicating successful storage
        """
        try:
            result = self.storage.put_memory(text=text, id=id, tags=tags)
            logger().info(f"Successfully stored memory with id={result['id']}")
            return result
        except Exception as e:
            logger().error(f"Failed to store memory: {str(e)}")
            return {
                "error": str(e),
                "stored": False
            }


class SearchMemory(Tools):
    """Find stored memories relevant to a query using semantic search."""
    
    For: ClassVar[Annotated[ToolType, ToolType.FOR_SELF | ToolType.FOR_DATA_PATTERNS]]
    
    def __init__(self):
        super().__init__()
        self._storage = None
    
    @property
    def storage(self) -> MemoryStorage:
        """Lazy initialization of storage."""
        if self._storage is None:
            self._storage = MemoryStorage()
        return self._storage
    
    async def invoke(
        self,
        query: str,
        top_k: int = 5,
        tag_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        # Handle tag_filter if it's passed as a JSON string
        if isinstance(tag_filter, str):
            import json
            try:
                tag_filter = json.loads(tag_filter)
            except json.JSONDecodeError:
                tag_filter = None
        """ALWAYS search memories FIRST when working with Dremio data, before schema discovery or SQL generation.

        This tool finds stored memories relevant to the user's question. Use it:
        - IMMEDIATELY when user asks about any data analysis, metrics, or queries
        - BEFORE using GetSchemaOfTable, GetUsefulSystemTableNames, or RunSqlQuery
        - When user mentions specific domains (e.g., 'quarterbacks', 'revenue', 'customers')
        - For any question that might have been answered before

        Search for relevant approaches, query patterns, table names, and business logic that have been
        previously stored. This prevents re-discovering schemas and ensures consistent methodology.
        
        Args:
            query: Search query to find relevant memories (e.g., "ARR definition", "revenue model")
            top_k: Maximum number of results to return (default: 5)
            tag_filter: Optional list of tags to filter results (e.g., ["metric", "finance"])
            
        Returns:
            Dictionary containing:
            - matches: List of matching memories with id, score, text_preview, and tags
            - query_embedding_dim: Dimension of the embedding space used for search
        """
        try:
            result = self.storage.search_memory(query=query, top_k=top_k, tag_filter=tag_filter)
            logger().info(f"Search for '{query}' returned {len(result['matches'])} matches")
            return result
        except Exception as e:
            import traceback
            logger().error(f"Failed to search memories: {str(e)}")
            logger().error(f"Traceback: {traceback.format_exc()}")
            return {
                "error": str(e),
                "matches": [],
                "query_embedding_dim": 0
            }


class GetMemory(Tools):
    """Retrieve the full text of a specific memory by its ID."""
    
    For: ClassVar[Annotated[ToolType, ToolType.FOR_SELF | ToolType.FOR_DATA_PATTERNS]]
    
    def __init__(self):
        super().__init__()
        self._storage = None
    
    @property
    def storage(self) -> MemoryStorage:
        """Lazy initialization of storage."""
        if self._storage is None:
            self._storage = MemoryStorage()
        return self._storage
    
    async def invoke(self, id: str) -> Dict[str, Any]:
        """Retrieve the full text and details of a specific memory by its unique ID.
        
        Use this when you have a memory ID from search results and need the complete text
        content, not just the preview.
        
        Args:
            id: The unique identifier of the memory to retrieve
            
        Returns:
            Dictionary containing:
            - id: The memory identifier
            - text: The full text content
            - tags: List of associated tags
            - ts: Timestamp when the memory was created/updated
            
            Returns None if the memory is not found.
        """
        try:
            result = self.storage.get_memory(id=id)
            if result:
                logger().info(f"Retrieved memory with id={id}")
                return result
            else:
                logger().warning(f"Memory not found with id={id}")
                return {
                    "error": f"Memory not found with id={id}",
                    "id": id
                }
        except Exception as e:
            logger().error(f"Failed to retrieve memory {id}: {str(e)}")
            return {
                "error": str(e),
                "id": id
            }


class SearchRelevantMemories(Tools):
    """Search for any relevant memories before starting work on a user question."""

    For: ClassVar[Annotated[ToolType, ToolType.FOR_SELF | ToolType.FOR_DATA_PATTERNS]]

    def __init__(self):
        super().__init__()
        self._storage = None

    @property
    def storage(self) -> MemoryStorage:
        """Lazy initialization of storage."""
        if self._storage is None:
            self._storage = MemoryStorage()
        return self._storage

    async def invoke(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """Search for relevant memories before starting any analysis or task.

        RECOMMENDED: Use this tool FIRST for any user question to check if there are
        relevant stored approaches, methodologies, or previous solutions.

        This is especially important for:
        - Data analysis questions
        - Requests involving specific domains (sports, finance, etc.)
        - Questions about "top X" or rankings
        - Any query that might have been solved before

        Args:
            question: The user's question or request
            top_k: Maximum number of relevant memories to return (default: 3)

        Returns:
            Dictionary containing:
            - matches: List of relevant memories that might help answer the question
            - has_relevant_memories: Boolean indicating if useful memories were found
            - suggestion: Text suggestion on how to use the found memories
        """
        try:
            # Extract key terms from the question for search
            import re

            # Look for key domain terms and concepts
            search_terms = []

            # Extract potential domain terms
            words = re.findall(r'\b\w+\b', question.lower())

            # Common data analysis terms
            data_terms = ['quarterback', 'qb', 'player', 'team', 'game', 'season', 'stats',
                         'revenue', 'customer', 'sales', 'metric', 'analysis', 'top', 'ranking',
                         'performance', 'data', 'query', 'sql', 'table', 'database']

            # Find relevant terms in the question
            relevant_terms = [word for word in words if word in data_terms or len(word) > 6]

            if relevant_terms:
                search_query = ' '.join(relevant_terms[:5])  # Use top 5 relevant terms
            else:
                search_query = question[:100]  # Use first 100 chars as fallback

            result = self.storage.search_memory(query=search_query, top_k=top_k)

            matches = result.get('matches', [])
            has_relevant = len(matches) > 0 and any(match['score'] > 0.2 for match in matches)

            suggestion = ""
            if has_relevant:
                high_score_matches = [m for m in matches if m['score'] > 0.3]
                if high_score_matches:
                    suggestion = f"Found {len(high_score_matches)} highly relevant memories. Consider using the stored approach from '{high_score_matches[0]['id']}' before starting fresh analysis."
                else:
                    suggestion = f"Found {len(matches)} potentially relevant memories. Review them to see if any apply to your current question."
            else:
                suggestion = "No highly relevant memories found. Proceed with normal analysis, but consider storing your approach for future use."

            logger().info(f"Searched memories for question '{question[:50]}...', found {len(matches)} matches")

            return {
                "matches": matches,
                "has_relevant_memories": has_relevant,
                "suggestion": suggestion,
                "search_query_used": search_query
            }
        except Exception as e:
            logger().error(f"Failed to search relevant memories: {str(e)}")
            return {
                "matches": [],
                "has_relevant_memories": False,
                "suggestion": "Memory search failed. Proceed with normal analysis.",
                "error": str(e)
            }


class ListMemories(Tools):
    """List all stored memories with basic information."""
    
    For: ClassVar[Annotated[ToolType, ToolType.FOR_SELF | ToolType.FOR_DATA_PATTERNS]]
    
    def __init__(self):
        super().__init__()
        self._storage = None
    
    @property
    def storage(self) -> MemoryStorage:
        """Lazy initialization of storage."""
        if self._storage is None:
            self._storage = MemoryStorage()
        return self._storage
    
    async def invoke(self, limit: int = 100) -> Dict[str, Any]:
        """List all stored memories with basic information (preview, tags, timestamp).
        
        Useful for understanding what memories are available and managing the memory store.
        
        Args:
            limit: Maximum number of memories to return (default: 100)
            
        Returns:
            Dictionary containing:
            - memories: List of memory summaries with id, text_preview, tags, and timestamp
            - count: Total number of memories returned
        """
        try:
            memories = self.storage.list_memories(limit=limit)
            logger().info(f"Listed {len(memories)} memories")
            return {
                "memories": memories,
                "count": len(memories)
            }
        except Exception as e:
            logger().error(f"Failed to list memories: {str(e)}")
            return {
                "error": str(e),
                "memories": [],
                "count": 0
            }
