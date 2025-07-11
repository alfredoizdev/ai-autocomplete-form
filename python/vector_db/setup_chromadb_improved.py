import json
import chromadb
from chromadb.utils import embedding_functions
import uuid
import os
from pathlib import Path
from datetime import datetime
import time

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

def setup_chromadb():
    """Initialize ChromaDB client and create collection"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Setting up ChromaDB...")
    
    # Use PersistentClient for local storage (faster and more reliable)
    db_path = Path(__file__).parent.parent / "chroma_db"
    client = chromadb.PersistentClient(path=str(db_path))
    
    # Use sentence transformers for better quality embeddings
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # Fast and good quality
    )
    
    # Create or get collection
    try:
        # Delete existing collection if it exists (for clean setup)
        client.delete_collection(name="bio_embeddings_v2")
        print("Deleted existing collection")
    except:
        pass
    
    collection = client.create_collection(
        name="bio_embeddings_v2",
        embedding_function=sentence_transformer_ef
    )
    
    print("Created new collection: bio_embeddings_v2")
    return client, collection

def index_bios(collection, bio_file_path):
    """Load and index bios into ChromaDB with progress tracking"""
    start_time = time.time()
    
    # Load bio data
    with open(bio_file_path, 'r') as f:
        bios = json.load(f)
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loaded {len(bios)} bios from {bio_file_path}")
    
    # Prepare data for ChromaDB
    documents = []
    metadatas = []
    ids = []
    
    for i, bio_text in enumerate(bios):
        # Skip very short bios
        if len(bio_text) < 20:
            continue
            
        documents.append(bio_text)
        metadatas.append({
            "index": i, 
            "length": len(bio_text),
            "word_count": len(bio_text.split())
        })
        ids.append(str(uuid.uuid4()))
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Prepared {len(documents)} bios for indexing")
    
    # Add to ChromaDB in batches
    batch_size = 50  # Smaller batch size for stability
    total_indexed = 0
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting batch indexing...")
    
    for i in range(0, len(documents), batch_size):
        batch_start = time.time()
        
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        try:
            collection.add(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids
            )
            
            total_indexed += len(batch_docs)
            batch_time = time.time() - batch_start
            
            # Progress update every 10 batches
            if (i // batch_size) % 10 == 0:
                elapsed = time.time() - start_time
                rate = total_indexed / elapsed
                remaining = (len(documents) - total_indexed) / rate
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress: {total_indexed}/{len(documents)} "
                      f"({total_indexed/len(documents)*100:.1f}%) - "
                      f"Speed: {rate:.1f} docs/sec - "
                      f"ETA: {remaining/60:.1f} minutes")
        
        except Exception as e:
            print(f"Error indexing batch {i//batch_size}: {e}")
            continue
    
    total_time = time.time() - start_time
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Successfully indexed {total_indexed} bios in {total_time/60:.1f} minutes")
    
    # Save indexing stats
    stats = {
        "total_bios": len(bios),
        "indexed_bios": total_indexed,
        "indexing_time_seconds": total_time,
        "timestamp": datetime.now().isoformat()
    }
    
    stats_path = Path(__file__).parent.parent / "chroma_db" / "indexing_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return collection

def test_search(collection):
    """Test the vector search with sample queries"""
    test_queries = [
        "I am a software engineer",
        "Looking for friends who",
        "My hobbies include",
        "I enjoy outdoor activities",
        "Looking for couples to",
        "We are a couple who"
    ]
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === Testing Vector Search ===")
    for query in test_queries:
        try:
            results = collection.query(
                query_texts=[query],
                n_results=3
            )
            
            print(f"\nQuery: '{query}'")
            if results['documents'][0]:
                first_result = results['documents'][0][0]
                print(f"Top result: {first_result[:100]}...")
                print(f"Distance: {results['distances'][0][0]:.4f}")
        except Exception as e:
            print(f"Error querying '{query}': {e}")

def main():
    """Main setup function"""
    print("="*60)
    print("ChromaDB Setup for Bio Autocomplete (Improved)")
    print("="*60)
    
    # Setup ChromaDB
    client, collection = setup_chromadb()
    
    # Path to bio.json
    bio_file_path = PROJECT_ROOT / "data" / "bio.json"
    
    # Index bios
    collection = index_bios(collection, bio_file_path)
    
    # Test search functionality
    test_search(collection)
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ✅ ChromaDB setup complete!")
    print("Vector database is ready for use.")
    print(f"Collection name: bio_embeddings_v2")
    print(f"Total documents: {collection.count()}")

if __name__ == "__main__":
    main()