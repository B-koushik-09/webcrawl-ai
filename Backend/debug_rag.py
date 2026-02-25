import sys
import os
import time

# Add current dir to path
sys.path.append(os.getcwd())

from config import RAG_CONFIG
from modules.embeddings import KnowledgeIndex
from modules.rag_engine import initialize_rag_engine

def debug_retrieval():
    print("Loading Knowledge Index...")
    index = KnowledgeIndex()
    if not index.load_index():
        print("[ERROR] Failed to load index. Is it built?")
        return

    print(f"[OK] Index loaded with {index.index.ntotal} items.")
    
    rag = initialize_rag_engine(index)
    
    query = "How many B.Tech and M.Tech programs does VNRVJIET offer?"
    print(f"\nQuery: '{query}'")
    print(f"Config: top_k={RAG_CONFIG['top_k']}")
    
    # Run the query
    t0 = time.time()
    response = rag.query(query)
    dt = time.time() - t0
    
    print(f"\nQuery completed in {dt:.2f}s")
    print(f"Confidence: {response.confidence}")
    print(f"Answer: {response.answer[:100]}...")
    
    print("\n=== Retrieved Chunks ===")
    found_table_start = False
    found_table_end = False
    
    for i, cit in enumerate(response.citations):
        content = cit.snippet # Snippet might be truncated in citation object? 
        # Actually RAGResponse stores full context in raw_context, but citation has snippets.
        # Let's peek at the actual content if possible.
        
        print(f"\n[Chunk {i+1}] Source: {cit.source_name} (Score: {cit.relevance_score:.3f})")
        print("-" * 40)
        print(cit.snippet[:300].replace('\n', ' '))
        
        # Check for specific rows
        if "| 1 |" in cit.snippet and "Artificial Intelligence" in cit.snippet:
            print(">>> FOUND START OF B.TECH TABLE <<<")
            found_table_start = True
        
        if "| 17 |" in cit.snippet and "Robotics" in cit.snippet:
            print(">>> FOUND END OF B.TECH TABLE <<<")
            found_table_end = True
            
    if found_table_start and found_table_end:
        print("\n[SUCCESS] Both start and end of table found!")
    elif found_table_start:
         print("\n[PARTIAL] Found start of table but missing end.")
    elif found_table_end:
         print("\n[PARTIAL] Found end of table but missing start.")
    else:
         print("\n[FAILURE] Table not found in top chunks.")

if __name__ == "__main__":
    debug_retrieval()
