"""
Debug retrieval to find where 'August 2006' comes from.
"""
from modules.embeddings import KnowledgeIndex
from modules.rag_engine import initialize_rag_engine

def debug_query():
    # Load index
    print("Loading index...")
    index = KnowledgeIndex()
    if not index.load_index():
        print("Failed to load index!")
        return
        
    engine = initialize_rag_engine(index)
    
    query = "When was VNRVJIET established?"
    
    print(f"\nQuery: {query}")
    print("=" * 60)
    
    # 1. Inspect Retrieval
    print("Retrieving chunks...")
    results = index.search(query, top_k=10)
    
    for i, (item, score) in enumerate(results):
        print(f"\n[Chunk {i+1}] Score: {score:.3f}")
        print(f"Source: {item.source_name} ({item.source_type})")
        print(f"Content:\n{item.content}")
        print("-" * 40)
        
        if "2006" in item.content:
            print(">>> FOUND '2006' IN THIS CHUNK! <<<")

if __name__ == "__main__":
    debug_query()
