import asyncio
import os
import sys

# Add backend directory to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.rag_engine import get_rag_engine
from modules.embeddings import get_knowledge_index

async def debug_retrieval():
    print("Initializing components...")
    index = get_knowledge_index()
    index.load_status()
    
    engine = get_rag_engine()
    engine.set_index(index)
    
    query = "Who is the principal of the college"
    
    print("\n[1] Testing Raw Context Retrieval Matrix:")
    try:
        results = engine.get_relevant_context(query)
        print(f"Found {len(results)} chunks.")
        for i, (item, score) in enumerate(results):
            print(f"\n--- Result {i+1} (Score: {score:.3f}) ---")
            print(f"File: {item.source_name}")
            print(f"Type: {item.doc_type}")
            print(f"Content snippet: {item.content[:150]}...")
    except Exception as e:
        print(f"Error during retrieval: {e}")
        
    print("\n[2] Testing Full AI Generation Process:")
    try:
        response = engine.query(query)
        print("\nAnswer Output:")
        print(response.answer)
        print(f"\nGrounded: {response.grounded}")
        print(f"Confidence: {response.confidence}")
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    asyncio.run(debug_retrieval())
