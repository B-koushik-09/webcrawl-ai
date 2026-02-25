
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from modules.embeddings import KnowledgeIndex
from modules.rag_engine import RAGEngine

def test_placement_queries():
    print("Initializing Knowledge Index...")
    index = KnowledgeIndex()
    if not index.load():
        print("Error: Could not load index.")
        return

    engine = RAGEngine(index)
    
    queries = [
        "How many students were placed in IT department in 2022-23?",
        "What is the average placement CTC for CSE?",
        "Which companies visit for campus placements?",
        "Who got placed in Amazon from IT?",
        "pacement data for eee"
    ]
    
    for query_text in queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query_text}")
        print(f"{'='*80}")
        
        response = engine.query(query_text)
        
        print(f"\nANSWER: {response.answer}")
        print(f"CONFIDENCE: {response.confidence:.2f}")
        print(f"GROUNDED: {response.grounded}")
        
        print("\nTOP CITATIONS:")
        for i, cit in enumerate(response.citations[:3], 1):
            print(f"{i}. {cit.source_name} (Score: {cit.relevance_score:.3f})")
            print(f"   Snippet: {cit.snippet[:150]}...")
            
        if not response.grounded:
            print("\nLOW CONFIDENCE / REJECTED CHUNKS:")
            # Manual search to see what was retrieved before verification
            results = index.search(query_text, top_k=5, min_similarity=0.3)
            for i, (item, score) in enumerate(results, 1):
                print(f"[RETR {i}] Score: {score:.3f} | Source: {item.source_name}")
                print(f"      Content: {item.content[:200]}...")

if __name__ == "__main__":
    test_placement_queries()
