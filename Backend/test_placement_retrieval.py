
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add backend to path
sys.path.append(str(Path(__file__).parent))
load_dotenv()

from modules.embeddings import KnowledgeIndex
from modules.rag_engine import RAGEngine

def test_placement_retrieval():
    print("Initializing Knowledge Index...")
    index = KnowledgeIndex()
    if not index.load():
        print("Error: Could not load index.")
        return

    engine = RAGEngine(index)
    
    # These queries previously failed or returned "not found"
    queries = [
        "What is the highest package in 2022-23?",
        "What are the placement statistics for IT department?",
        "Which companies offered packages above 20 LPA?",
        "What is the average CTC for CSE?",
        "How many students were placed in Amazon?"
    ]
    
    for query_text in queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query_text}")
        print(f"{'='*80}")
        
        response = engine.query(query_text)
        
        print(f"\nANSWER:\n{response.answer}")
        print(f"\nCONFIDENCE: {response.confidence:.2f}")
        print(f"GROUNDED: {response.grounded}")
        print(f"NUM CITATIONS: {len(response.citations)}")
        
        if response.citations:
            print("\nSOURCES:")
            for i, cit in enumerate(response.citations[:3], 1):
                print(f"{i}. {cit.source_name} (Score: {cit.relevance_score:.3f})")

if __name__ == "__main__":
    test_placement_retrieval()
