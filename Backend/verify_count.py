
import sys
import os
import time

sys.path.append(os.getcwd())

from modules.embeddings import KnowledgeIndex
from modules.rag_engine import initialize_rag_engine

def test_program_counts():
    print("Loading Knowledge Index...")
    index = KnowledgeIndex()
    if not index.load_index():
        print("[ERROR] Failed to load index.")
        return

    rag = initialize_rag_engine(index)
    
    query = "How many programmes does VNRVJIET offer?"
    print(f"\nQuery: '{query}'")
    
    response = rag.query(query)
    
    print("\n=== Answer ===")
    print(response.answer)
    print("\n=== Top Citations ===")
    found_summary = False
    
    for i, cit in enumerate(response.citations[:5]):
        print(f"\n[Chunk {i+1}] Source: {cit.source_name} (Score: {cit.relevance_score:.3f})")
        print(f"Snippet: {cit.snippet[:150]}...")
        
        if "14 B.Tech" in cit.snippet and "13 M.Tech" in cit.snippet:
            print(">>> SUCCESS: Found the summary chunk!")
            found_summary = True
    
    if found_summary:
        print("\nTEST PASSED: Summary chunk retrieved.")
    else:
        print("\nTEST FAILED: Summary chunk NOT retrieved.")

if __name__ == "__main__":
    test_program_counts()
