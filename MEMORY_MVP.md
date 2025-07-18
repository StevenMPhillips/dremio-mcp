# Dremio MCP Memory MVP

## Overview

The Dremio MCP Memory MVP adds persistent memory capabilities to the Dremio MCP server, allowing AI assistants to store and retrieve business knowledge across sessions. This ensures consistent use of business definitions, metric calculations, and data glossaries.

## Features

### Core Functionality
- **Persistent Storage**: Memories are stored in a local DuckDB database that persists across sessions
- **Vector Similarity Search**: TF-IDF vectorization with cosine similarity for semantic understanding
- **Tag-based Organization**: Categorize memories with tags for better organization and filtering
- **Full-text Retrieval**: Get complete memory content by ID
- **Automatic Caching**: In-memory vector cache for fast search performance

### Memory Tools (Priority Order)

#### 1. `SearchRelevantMemories` ⭐ **NEW - Use First**
Search for any relevant memories before starting work on a user question.

**When to use:** FIRST tool for any user question, especially data analysis requests.

**Parameters:**
- `question` (required): The user's question or request
- `top_k` (optional): Maximum results to return (default: 3)

**Example:**
```bash
uv run dremio-mcp-server tools invoke -t SearchRelevantMemories \
  question="show me top 10 quarterbacks of 2023"
```

**Key Benefits:**
- Automatically extracts relevant terms from user questions
- Provides guidance on whether to use stored approaches
- Prevents redundant schema discovery when methodology exists

#### 2. `SearchMemory`
Find stored memories relevant to specific queries using semantic search.

**When to use:** When you need domain-specific knowledge or methodologies.

**Parameters:**
- `query` (required): Search query string
- `top_k` (optional): Maximum results to return (default: 5)
- `tag_filter` (optional): Filter results by tags

**Example:**
```bash
uv run dremio-mcp-server tools invoke -t SearchMemory \
  query="quarterback EPA methodology" \
  tag_filter='["nfl","methodology"]'
```

#### 3. `PutMemory`
Store enterprise memories (metric definitions, glossary notes) for future retrieval.

**Parameters:**
- `text` (required): The text content to store
- `id` (optional): Unique identifier (UUID generated if not provided)
- `tags` (optional): List of tags for categorization

**Example:**
```bash
uv run dremio-mcp-server tools invoke -t PutMemory \
  text="ARR is NOT simply MRR * 12. Our calculation method: ..." \
  id="metric:ARR" \
  tags='["metric","finance","ARR","subscription"]'
```

#### 2. `SearchMemory`
Find stored memories relevant to a query using semantic search.

**Parameters:**
- `query` (required): Search query string
- `top_k` (optional): Maximum results to return (default: 5)
- `tag_filter` (optional): Filter results by tags

**Example:**
```bash
uv run dremio-mcp-server tools invoke -t SearchMemory \
  query="ARR definition" \
  top_k=3 \
  tag_filter='["finance"]'
```

#### 3. `GetMemory`
Retrieve the full text and details of a specific memory by ID.

**Parameters:**
- `id` (required): The unique identifier of the memory

**Example:**
```bash
uv run dremio-mcp-server tools invoke -t GetMemory id="metric:ARR"
```

#### 4. `ListMemories`
List all stored memories with basic information.

**Parameters:**
- `limit` (optional): Maximum number of memories to return (default: 100)

**Example:**
```bash
uv run dremio-mcp-server tools invoke -t ListMemories limit=10
```

## Configuration

Memory settings can be configured in your `config.yaml`:

```yaml
memory:
  db_path: "~/.config/dremioai/memory.db"  # Database file location
  embedding_model: "all-MiniLM-L6-v2"     # Embedding model (for future use)
  embedding_dim: 384                       # Embedding dimensions
  max_preview_chars: 240                   # Preview text length
```

## Sample Data

The system comes with pre-seeded business memories including:

- **ARR Definition**: Proper calculation method (not MRR × 12)
- **MRR Definition**: Monthly recurring revenue calculation
- **ARR vs MRR**: Key differences and when to use each
- **Churn Rate**: Customer churn calculation methodology
- **Customer Success DB**: Performance optimization notes
- **Monthly Revenue Close**: Process documentation

## Usage Patterns

### For AI Assistants (Memory-First Workflow)

#### 🎯 **RECOMMENDED WORKFLOW:**
1. **Start with SearchRelevantMemories**: For ANY user question
   ```
   SearchRelevantMemories(question="show me top 10 quarterbacks of 2023")
   ```

2. **Get full methodology**: If relevant memories found
   ```
   GetMemory(id="quarterback_analysis_methodology")
   ```

3. **Search for specifics**: If you need domain knowledge
   ```
   SearchMemory(query="NFL quarterback EPA rushing", tag_filter=["methodology"])
   ```

4. **THEN proceed to schema discovery**: Only if memories don't provide guidance
   ```
   GetSchemaOfTable(table_name="myData.pbp")
   ```

5. **Store new approaches**: After successful analysis
   ```
   PutMemory(text="...", tags=["nfl", "methodology"])
   ```

#### 🚫 **AVOID THIS OLD PATTERN:**
- ❌ User asks question → Immediate schema discovery → Trial and error
- ✅ User asks question → Search memories FIRST → Use stored approach

### For Users
1. **Build knowledge base**: Store important business definitions
2. **Maintain consistency**: Ensure all team members use the same calculations
3. **Document processes**: Keep track of data procedures and best practices

## Demo

Run the demo scripts to see the memory system in action:

### Basic Functionality Demo
```bash
./demo_simple.sh
```

### Vector Similarity Search Demo
```bash
./demo_vector_search.sh
```

### Memory-First Workflow Demo
```bash
./demo_memory_first.sh
```

These demonstrate:
- Vector similarity search with semantic understanding
- Memory-first workflow for data analysis
- Tag-based filtering with vector search
- Listing, storing, and retrieving memories
- Quality comparison with different query types

## Technical Implementation

### Storage
- **Database**: DuckDB for lightweight, serverless persistence
- **Schema**: Simple table structure with id, text, tags, and timestamps
- **Location**: `~/.config/dremioai/memory.db` by default

### Search Algorithm
- **TF-IDF Vectorization**: Uses scikit-learn's TfidfVectorizer with 1000 features
- **N-gram Features**: Includes unigrams and bigrams for better semantic understanding
- **Cosine Similarity**: Proper vector similarity scoring (0.0 to 1.0 range)
- **Stop Word Removal**: Filters out common English stop words
- **Vector Caching**: In-memory cache for fast repeated searches

### Integration
- **MCP Tools**: Seamlessly integrated with existing Dremio MCP tools
- **Tool Types**: Available for `FOR_SELF` and `FOR_DATA_PATTERNS` modes
- **Error Handling**: Graceful fallbacks and informative error messages

## Future Enhancements

1. **Advanced Embeddings**: Replace text similarity with proper semantic embeddings
2. **Structured Metrics**: Support parameterized metric definitions
3. **Access Control**: Role-based memory visibility
4. **Sync/Backup**: Cloud storage integration for team sharing
5. **Auto-categorization**: AI-powered tag suggestions
6. **Metric Validation**: Detect inconsistent metric usage

## Getting Started

1. **Install Dependencies**: Already included with `uv add duckdb numpy`
2. **Seed Sample Data**: Run `uv run python -m dremioai.memory.seed`
3. **Test Tools**: Use `uv run dremio-mcp-server tools list -m FOR_SELF`
4. **Run Demo**: Execute `./demo_simple.sh`
5. **Integrate with MCP Client**: Use with Claude Desktop, ChatGPT, or any MCP client

The memory system is now ready for production use and will help ensure consistent business knowledge across all AI interactions!
