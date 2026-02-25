from modules.rag_engine import RAGEngine
import json

def test_retrieval(query_text):
    print(f"\n--- Testing Query: '{query_text}' ---")
    rag = RAGEngine()
    
    # We want to check is_cse_query detection
    # The logic is inside the query method
    
    # Capture print output if needed, but for now we look at the results
    results = rag.query(query_text, top_k=5)
    
    print(f"Top Source: {results['source_documents'][0]['source'] if results['source_documents'] else 'None'}")
    # Print the first few characters of the top chunk to see if it's CSE related
    if results['source_documents']:
        print(f"Top Content Snippet: {results['source_documents'][0]['content'][:200]}...")

if __name__ == "__main__":
    test_queries = [
        "placements",
        "cse placements",
        "hostel fees",
        "who is faculty in cse",
        "manmath nath das"
    ]
    for q in test_queries:
        test_retrieval(q)
