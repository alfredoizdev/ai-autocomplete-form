import chromadb
from chromadb.utils import embedding_functions
import json
from pathlib import Path

def check_chromadb_status():
    """Check the status of ChromaDB collections"""
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # List all collections
        collections = client.list_collections()
        print(f"Found {len(collections)} collections:")
        for col in collections:
            print(f"  - {col.name}")
        
        # Check bio_embeddings collection
        try:
            collection = client.get_collection(
                name="bio_embeddings",
                embedding_function=embedding_functions.DefaultEmbeddingFunction()
            )
            print(f"\nCollection 'bio_embeddings':")
            print(f"  - Document count: {collection.count()}")
            
            # Test a sample query
            results = collection.query(
                query_texts=["Looking for"],
                n_results=3
            )
            
            print(f"  - Sample query results: {len(results['documents'][0])} documents found")
            if results['documents'][0]:
                print(f"  - First result preview: {results['documents'][0][0][:100]}...")
        except Exception as e:
            print(f"  - Error accessing bio_embeddings: {e}")
        
        # Check bio_embeddings_v2 collection if it exists
        try:
            collection_v2 = client.get_collection(
                name="bio_embeddings_v2",
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            )
            print(f"\nCollection 'bio_embeddings_v2':")
            print(f"  - Document count: {collection_v2.count()}")
        except Exception as e:
            print(f"\nCollection 'bio_embeddings_v2' not found or error: {e}")
        
        # Check indexing stats if available
        stats_path = Path("./chroma_db/indexing_stats.json")
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            print(f"\nIndexing statistics:")
            print(f"  - Total bios: {stats.get('total_bios', 'N/A')}")
            print(f"  - Indexed bios: {stats.get('indexed_bios', 'N/A')}")
            print(f"  - Timestamp: {stats.get('timestamp', 'N/A')}")
        
    except Exception as e:
        print(f"Error checking ChromaDB: {e}")

if __name__ == "__main__":
    check_chromadb_status()