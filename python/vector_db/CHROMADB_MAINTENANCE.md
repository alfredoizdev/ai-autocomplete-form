# ChromaDB Maintenance Guide

## Overview
This guide helps maintain the ChromaDB vector database used for bio autocomplete functionality in hybrid mode.

## Common Issues

### 1. Corrupted or Incomplete Data
**Symptoms:**
- ChromaDB has fewer documents than expected (should have ~4,994 from bio.json)
- Autocomplete returns empty results
- API returns errors when accessing vector search

**Solution:**
```bash
# 1. Check current status
cd python/vector_db
python3 check_chromadb_status.py

# 2. If document count is incorrect, re-index:
python3 setup_chromadb.py
```

### 2. Data Format Requirements
- **Input**: bio.json must be an array of strings
- **Minimum Length**: Bios shorter than 20 characters are filtered out
- **Format**: Plain text strings (no nested objects)

### 3. Re-indexing Procedure
To completely rebuild the ChromaDB:

```bash
cd python/vector_db

# This will:
# - Delete existing collection
# - Create new collection
# - Index all bios from data/bio.json
# - Test with sample queries
python3 setup_chromadb.py
```

### 4. Testing the Database
After re-indexing, verify it's working:

```bash
# Check document count
python3 check_chromadb_status.py

# Test API endpoint
curl -X POST http://localhost:8001/api/autocomplete/hybrid \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Looking for couples who"}'
```

## Important Notes
- The API uses `bio_embeddings` collection (not `bio_embeddings_v2`)
- ChromaDB data is stored in `python/vector_db/chroma_db/`
- The hybrid API expects field name `prompt` (not `partial_text`)
- Re-indexing takes about 1-2 minutes for ~5,000 bios

## Monitoring
Check these regularly:
1. Document count should match bio.json entry count
2. API response time should be < 200ms for vector search
3. Hybrid endpoint should return 2-3 suggestions