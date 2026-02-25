
import sys
import os
import time

sys.path.append(os.getcwd())

from modules.embeddings import KnowledgeIndex
from modules.rag_engine import initialize_rag_engine

def test_btech_filtering():
    print("Loading Knowledge Index...")
    index = KnowledgeIndex()
    if not index.load_index():
        print("[ERROR] Failed to load index.")
        return

    rag = initialize_rag_engine(index)
    
    # Query specifically for B.Tech
    query = "How is admission to B.Tech courses conducted?"
    print(f"\nQuery: '{query}'")
    
    response = rag.query(query)
    
    print("\n=== Verified Chunks ===")
    mtech_count = 0
    btech_count = 0
    
    for i, cit in enumerate(response.citations):
        print(f"\n[Chunk {i+1}] Source: {cit.source_name}")
        print(f"Snippet: {cit.snippet[:100]}...")
        
        content_lower = (cit.source_name + " " + cit.snippet).lower()
        
        if "m.tech" in content_lower or "mtech" in content_lower:
            # Check if it also mentions B.Tech (valid mixed content)
            if "b.tech" not in content_lower and "btech" not in content_lower:
                print(">>> WARNING: Found M.Tech content in B.Tech query!")
                mtech_count += 1
            else:
                 print(">>> Info: Mixed content (valid)")
        
        if "b.tech" in content_lower or "btech" in content_lower:
            btech_count += 1

    print(f"\nSummary:")
    print(f"Total Citations: {len(response.citations)}")
    print(f"B.Tech specific: {btech_count}")
    print(f"M.Tech specific (Should be 0): {mtech_count}")
    
    if mtech_count == 0:
        print("\nSUCCESS: Filtering working correctly.")
    else:
        print("\nFAILURE: M.Tech content leaked through.")

if __name__ == "__main__":
    test_btech_filtering()
