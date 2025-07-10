import json
import chromadb
from chromadb.utils import embedding_functions
import uuid
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

def setup_chromadb():
    """Initialize ChromaDB client and create collection"""
    # Initialize ChromaDB client - use PersistentClient for local storage
    # This avoids API version issues with the Docker container
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Use default embedding function to avoid version conflicts
    # We'll use the default embedding function for now
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    
    # Create or get collection
    try:
        # Delete existing collection if it exists (for clean setup)
        client.delete_collection(name="bio_embeddings")
        print("Deleted existing collection")
    except:
        pass
    
    collection = client.create_collection(
        name="bio_embeddings",
        embedding_function=default_ef
    )
    
    print("Created new collection: bio_embeddings")
    return client, collection

def index_bios(collection, bio_file_path):
    """Load and index bios into ChromaDB"""
    # Load bio data
    with open(bio_file_path, 'r') as f:
        bios = json.load(f)
    
    print(f"Loaded {len(bios)} bios from {bio_file_path}")
    
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
    
    print(f"Prepared {len(documents)} bios for indexing (filtered out short ones)")
    
    # Add to ChromaDB in batches for efficiency
    batch_size = 100
    total_indexed = 0
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )
        
        total_indexed += len(batch_docs)
        print(f"Indexed {total_indexed}/{len(documents)} bios...")
    
    print(f"Successfully indexed {len(documents)} bios into ChromaDB")
    return collection

def test_search(collection):
    """Test the vector search with sample queries"""
    test_queries = [
        "I am a software engineer",
        "Looking for friends who",
        "My hobbies include",
        "I enjoy",
        "Looking for couples"
    ]
    
    print("\n=== Testing Vector Search ===")
    for query in test_queries:
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        
        print(f"\nQuery: '{query}'")
        print(f"Found {len(results['documents'][0])} results")
        
        if results['documents'][0]:
            first_result = results['documents'][0][0]
            # Show first 100 chars of the result
            print(f"Top result: {first_result[:100]}...")

def main():
    """Main setup function"""
    print("Setting up ChromaDB for bio autocomplete...")
    
    # Setup ChromaDB
    client, collection = setup_chromadb()
    
    # Path to bio.json
    bio_file_path = PROJECT_ROOT / "data" / "bio.json"
    
    # Index bios
    collection = index_bios(collection, bio_file_path)
    
    # Test search functionality
    test_search(collection)
    
    print("\n✅ ChromaDB setup complete!")
    print("Vector database is ready for use.")

if __name__ == "__main__":
    main()