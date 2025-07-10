import chromadb
from chromadb.utils import embedding_functions
import re
from typing import List, Optional

class BioVectorSearch:
    def __init__(self, chroma_path: str = "./chroma_db"):
        """Initialize the vector search client"""
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=chroma_path)
        
        # Use default embedding function
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        
        # Get the collection
        try:
            self.collection = self.client.get_collection(
                name="bio_embeddings",
                embedding_function=self.embedding_function
            )
            print(f"Loaded collection with {self.collection.count()} items")
        except Exception as e:
            print(f"Error loading collection: {e}")
            raise
    
    def search_similar_contexts(self, query_text: str, n_results: int = 5) -> List[str]:
        """
        Search for similar bio contexts based on query
        Returns relevant bio segments for context
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            # Extract just the documents
            if results and 'documents' in results and results['documents']:
                return results['documents'][0]
            return []
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def get_autocomplete_suggestions(self, partial_text: str, max_suggestions: int = 3) -> List[str]:
        """
        Get autocomplete suggestions based on partial text
        Returns list of possible completions
        """
        # Clean the input
        partial_text = partial_text.strip()
        if not partial_text:
            return []
        
        # Search for similar contexts
        similar_bios = self.search_similar_contexts(partial_text, n_results=10)
        
        if not similar_bios:
            return []
        
        # Extract relevant completions
        suggestions = []
        words = partial_text.split()
        
        # Get the last few words for better matching
        if len(words) > 3:
            search_phrase = " ".join(words[-3:])
        else:
            search_phrase = partial_text
        
        search_phrase_lower = search_phrase.lower()
        
        for bio in similar_bios:
            bio_lower = bio.lower()
            
            # Find where the search phrase appears in the bio
            index = bio_lower.find(search_phrase_lower)
            
            if index != -1:
                # Get the continuation after the match
                start_pos = index + len(search_phrase)
                continuation = bio[start_pos:].strip()
                
                if continuation:
                    # Extract next few words
                    next_words = continuation.split()[:5]
                    if next_words:
                        suggestion = " ".join(next_words)
                        
                        # Clean up the suggestion
                        suggestion = self._clean_suggestion(suggestion)
                        
                        if suggestion and suggestion not in suggestions:
                            suggestions.append(suggestion)
                            
                            if len(suggestions) >= max_suggestions:
                                break
            else:
                # If exact match not found, look for partial matches
                words_in_bio = bio_lower.split()
                query_words = search_phrase_lower.split()
                
                # Find if the last word of query partially matches any word
                if query_words:
                    last_query_word = query_words[-1]
                    for i, word in enumerate(words_in_bio):
                        if word.startswith(last_query_word) and i < len(words_in_bio) - 1:
                            # Get the rest of the word and following words
                            rest_of_word = bio.split()[i][len(last_query_word):]
                            following_words = bio.split()[i+1:i+5]
                            
                            if rest_of_word:
                                suggestion = rest_of_word + " " + " ".join(following_words)
                            else:
                                suggestion = " ".join(following_words)
                            
                            suggestion = self._clean_suggestion(suggestion)
                            
                            if suggestion and suggestion not in suggestions:
                                suggestions.append(suggestion)
                                if len(suggestions) >= max_suggestions:
                                    break
        
        return suggestions[:max_suggestions]
    
    def _clean_suggestion(self, text: str) -> str:
        """Clean up suggestion text"""
        # Remove extra whitespace
        text = " ".join(text.split())
        
        # Remove incomplete sentences at the end
        if text.endswith((".", "!", "?")):
            return text
        
        # If there's punctuation in the middle, cut at the last complete sentence
        last_punct = max(
            text.rfind("."),
            text.rfind("!"),
            text.rfind("?")
        )
        
        if last_punct > 0:
            return text[:last_punct + 1]
        
        return text

def test_vector_search():
    """Test the vector search functionality"""
    print("Testing BioVectorSearch...")
    
    search = BioVectorSearch()
    
    test_queries = [
        "I am a software engineer",
        "Looking for friends who",
        "My hobbies include",
        "I enjoy",
        "Looking for couples",
        "I work as a"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        suggestions = search.get_autocomplete_suggestions(query)
        print(f"Suggestions ({len(suggestions)} found):")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")

if __name__ == "__main__":
    test_vector_search()