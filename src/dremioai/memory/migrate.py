#!/usr/bin/env python3
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

"""
Migration script for switching between embedding types.
"""

import asyncio
import sys
from dremioai.memory.storage import MemoryStorage
from dremioai.config import settings
from dremioai.log import logger


async def migrate_to_openai():
    """Migrate existing memories to use OpenAI embeddings."""
    config = settings.instance().memory
    
    # Check if OpenAI API key is configured
    if not config.openai_api_key:
        print("❌ OpenAI API key not configured!")
        print("Please add your OpenAI API key to the config:")
        print("")
        print("memory:")
        print("  embedding_type: 'openai'")
        print("  openai_api_key: 'sk-your-key-here'")
        print("")
        return False
    
    print(f"🔄 Migrating to OpenAI embeddings ({config.openai_model})...")
    print(f"📍 Database: {config.db_path}")
    
    try:
        storage = MemoryStorage()
        
        # Force remigration to OpenAI embeddings
        storage.force_remigration()
        
        print("✅ Migration completed successfully!")
        
        # Test the new embeddings
        print("\n🧪 Testing new embeddings...")
        result = storage.search_memory("quarterback analysis", top_k=2)
        print(f"Found {len(result['matches'])} matches with OpenAI embeddings")
        for match in result['matches']:
            print(f"  - {match['id']}: {match['score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        logger().error(f"Migration error: {e}")
        return False


async def migrate_to_tfidf():
    """Migrate existing memories to use TF-IDF embeddings."""
    print("🔄 Migrating to TF-IDF embeddings...")
    
    try:
        storage = MemoryStorage()
        
        # Force remigration to TF-IDF embeddings
        storage.force_remigration()
        
        print("✅ Migration completed successfully!")
        
        # Test the new embeddings
        print("\n🧪 Testing new embeddings...")
        result = storage.search_memory("quarterback analysis", top_k=2)
        print(f"Found {len(result['matches'])} matches with TF-IDF embeddings")
        for match in result['matches']:
            print(f"  - {match['id']}: {match['score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        logger().error(f"Migration error: {e}")
        return False


async def show_status():
    """Show current embedding configuration and status."""
    config = settings.instance().memory
    
    print("📊 Memory System Status")
    print("=" * 50)
    print(f"Embedding Type: {config.embedding_type}")
    print(f"Database Path: {config.db_path}")
    
    if config.embedding_type == 'openai':
        print(f"OpenAI Model: {config.openai_model}")
        print(f"API Key Configured: {'✅' if config.openai_api_key else '❌'}")
    else:
        print(f"TF-IDF Model: {config.embedding_model}")
    
    try:
        storage = MemoryStorage()
        
        # Count memories
        import duckdb
        from pathlib import Path
        db_path = Path(config.db_path).expanduser()
        conn = duckdb.connect(str(db_path))
        
        total_memories = conn.execute("SELECT COUNT(*) FROM mem").fetchone()[0]
        memories_with_vectors = conn.execute("SELECT COUNT(*) FROM mem WHERE vector IS NOT NULL").fetchone()[0]
        
        print(f"Total Memories: {total_memories}")
        print(f"With Vectors: {memories_with_vectors}")
        print(f"Migration Needed: {'❌' if memories_with_vectors < total_memories else '✅'}")
        
        if memories_with_vectors > 0:
            # Check vector dimension
            sample_dim = conn.execute("SELECT array_length(vector) FROM mem WHERE vector IS NOT NULL LIMIT 1").fetchone()
            if sample_dim:
                print(f"Vector Dimension: {sample_dim[0]}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")


def main():
    """Main entry point for migration script."""
    if len(sys.argv) < 2:
        print("Usage: python -m dremioai.memory.migrate <command>")
        print("")
        print("Commands:")
        print("  status    - Show current configuration and status")
        print("  openai    - Migrate to OpenAI embeddings")
        print("  tfidf     - Migrate to TF-IDF embeddings")
        print("")
        return
    
    command = sys.argv[1].lower()
    
    if command == "status":
        asyncio.run(show_status())
    elif command == "openai":
        success = asyncio.run(migrate_to_openai())
        sys.exit(0 if success else 1)
    elif command == "tfidf":
        success = asyncio.run(migrate_to_tfidf())
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
