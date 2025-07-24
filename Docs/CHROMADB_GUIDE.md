# ChromaDB Guide

Complete guide to working with ChromaDB vector database in the AI Bio Autocomplete system.

## 🎯 Overview

ChromaDB is the vector database that powers the hybrid autocomplete mode. It stores embeddings of 4,000+ bio examples and enables semantic search for finding similar text.

### Key Benefits
- **Fast**: ~80ms average query time
- **Semantic**: Finds conceptually similar text, not just keyword matches
- **Scalable**: Handles millions of documents
- **Simple**: Easy Python API

## 🚀 Quick Start

### Check Database Status
```bash
cd python/vector_db
python -c "from vector_search import BioVectorSearch; bs = BioVectorSearch(); print(f'Documents: {bs.collection.count()}')"
```

### Rebuild Database
```bash
cd python/vector_db
rm -rf chroma_db  # Remove old data
python setup_chromadb.py
```

## 📊 Understanding ChromaDB

### How It Works
1. **Text → Embeddings**: Converts bio text into 384-dimensional vectors
2. **Storage**: Stores vectors with metadata in persistent database
3. **Search**: Finds nearest neighbors using cosine similarity
4. **Retrieval**: Returns similar documents with relevance scores

### Data Flow
```
User Input → Query Embedding → Vector Search → Similar Bios → Context for LLM
   "Looking for..."  →  [0.23, -0.15, ...]  →  ChromaDB  →  5 matches  →  Ollama
```

## 🔧 Setup and Configuration

### Initial Setup
```python
# setup_chromadb.py
import chromadb
from chromadb.utils import embedding_functions

# Create client with persistent storage
client = chromadb.PersistentClient(path="./chroma_db")

# Use sentence-transformers for embeddings
default_ef = embedding_functions.DefaultEmbeddingFunction()

# Create or get collection
collection = client.get_or_create_collection(
    name="bio_embeddings",
    embedding_function=default_ef
)
```

### Adding Data
```python
# Load bio data
with open("../../data/bio.json", "r") as f:
    bios = json.load(f)

# Clean and filter
filtered_bios = [
    bio for bio in bios 
    if 10 <= len(bio.split()) <= 100  # Word count filter
]

# Add to ChromaDB
collection.add(
    documents=filtered_bios,
    ids=[f"bio_{i}" for i in range(len(filtered_bios))],
    metadatas=[{"length": len(bio.split())} for bio in filtered_bios]
)
```

## 🔍 Search Operations

### Basic Search
```python
# vector_search.py
def search_similar_bios(query: str, n_results: int = 5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results['documents'][0]
```

### Advanced Search with Filtering
```python
# Search with metadata filters
results = collection.query(
    query_texts=[query],
    n_results=5,
    where={"length": {"$gte": 20, "$lte": 50}}  # 20-50 words
)
```

### Similarity Threshold
```python
# Get results with distance scores
results = collection.query(
    query_texts=[query],
    n_results=10,
    include=["documents", "distances"]
)

# Filter by similarity (lower distance = more similar)
threshold = 0.7
filtered = [
    doc for doc, dist in zip(results['documents'][0], results['distances'][0])
    if dist < threshold
]
```

## 🛠 Maintenance

### Check Database Health
```python
# check_chromadb.py
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("bio_embeddings")

print(f"Total documents: {collection.count()}")
print(f"Collection name: {collection.name}")

# Sample query
results = collection.query(query_texts=["Looking for"], n_results=3)
print(f"Sample results: {results['documents'][0][:3]}")
```

### Clean Corrupted Data
```bash
# If database is corrupted
cd python/vector_db
rm -rf chroma_db
python setup_chromadb.py

# Verify
python -c "from vector_search import BioVectorSearch; print(BioVectorSearch().collection.count())"
```

### Backup Database
```bash
# Create backup
cd python/vector_db
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz chroma_db/

# Restore from backup
tar -xzf chroma_backup_20241224.tar.gz
```

## 📈 Performance Optimization

### Indexing Optimization
```python
# Batch operations for better performance
BATCH_SIZE = 100

for i in range(0, len(documents), BATCH_SIZE):
    batch_docs = documents[i:i+BATCH_SIZE]
    batch_ids = [f"bio_{j}" for j in range(i, i+len(batch_docs))]
    
    collection.add(
        documents=batch_docs,
        ids=batch_ids
    )
```

### Query Optimization
```python
class BioVectorSearch:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_collection("bio_embeddings")
        self._cache = {}  # Simple cache
    
    def search_with_cache(self, query: str, n_results: int = 5):
        cache_key = f"{query}:{n_results}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        self._cache[cache_key] = results
        return results
```

### Memory Management
```python
# For large datasets, use iteration
def process_large_dataset(file_path: str):
    with open(file_path, 'r') as f:
        batch = []
        for i, line in enumerate(f):
            batch.append(line.strip())
            
            if len(batch) >= 100:
                # Process batch
                collection.add(documents=batch, ids=[...])
                batch = []
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "Collection not found"
```python
# Check if collection exists
collections = client.list_collections()
print([c.name for c in collections])

# Create if missing
if "bio_embeddings" not in [c.name for c in collections]:
    collection = client.create_collection("bio_embeddings")
```

#### 2. Slow Queries
```python
# Check collection size
print(f"Documents: {collection.count()}")

# If too large, consider:
# 1. Adding metadata filters
# 2. Reducing n_results
# 3. Implementing caching
```

#### 3. Disk Space Issues
```bash
# Check database size
du -sh python/vector_db/chroma_db

# Clean old data
rm -rf python/vector_db/chroma_db
python setup_chromadb.py
```

## 📊 Analytics

### Usage Statistics
```python
# Track search patterns
import json
from datetime import datetime

def log_search(query: str, results_count: int):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "results": results_count
    }
    
    with open("search_log.json", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
```

### Quality Metrics
```python
# Analyze search quality
def analyze_search_quality(test_queries: list):
    for query in test_queries:
        results = collection.query(
            query_texts=[query],
            n_results=5,
            include=["distances"]
        )
        
        avg_distance = sum(results['distances'][0]) / len(results['distances'][0])
        print(f"Query: {query[:30]}... Avg Distance: {avg_distance:.3f}")
```

## 🔄 Integration with API

### In api_server.py
```python
# How it's used in the API
vector_search = BioVectorSearch(chroma_path="../vector_db/chroma_db")

# Get context for LLM
contexts = vector_search.get_context_for_prompt(prompt, n_contexts=3)

# Get exact matches
suggestions = vector_search.get_autocomplete_suggestions(prompt, max_suggestions=2)
```

## 💡 Best Practices

### 1. Data Quality
- Filter training data (10-100 words)
- Remove duplicates
- Ensure grammatical correctness

### 2. Search Strategy
- Use 3-5 results for context
- Filter by relevance score
- Cache frequent queries

### 3. Maintenance
- Regular backups
- Monitor growth
- Rebuild periodically

### 4. Performance
- Batch operations
- Implement caching
- Use metadata filters

## 🚀 Advanced Usage

### Custom Embeddings
```python
from sentence_transformers import SentenceTransformer

# Use a different model
model = SentenceTransformer('all-MiniLM-L6-v2')

def custom_embedding_function(texts):
    return model.encode(texts).tolist()

collection = client.create_collection(
    name="bio_embeddings_custom",
    embedding_function=custom_embedding_function
)
```

### Hybrid Search
```python
# Combine vector search with keyword filtering
def hybrid_search(query: str, keywords: list):
    # Vector search
    vector_results = collection.query(query_texts=[query], n_results=20)
    
    # Filter by keywords
    filtered = []
    for doc in vector_results['documents'][0]:
        if any(kw.lower() in doc.lower() for kw in keywords):
            filtered.append(doc)
    
    return filtered[:5]
```

## 📚 Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Vector Database Concepts](https://www.pinecone.io/learn/vector-database/)

## 🎯 Key Takeaways

1. **ChromaDB is the brain** of hybrid mode - it finds relevant context
2. **Quality matters** - Clean data = better suggestions
3. **Fast queries** - 80ms average with proper setup
4. **Easy maintenance** - Simple Python API
5. **Scalable** - Handles growth gracefully

Remember: Good vector search makes great autocomplete!