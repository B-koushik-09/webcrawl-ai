import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from modules.rag_engine import get_rag_engine

def debug_query(query):
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    rag = get_rag_engine()
    
    # We call query and intercept the output if we could, 
    # but let's just use the query() method and check what it does.
    # To see the expanded query, we'd need to mock or edit rag_engine.
    # Let's just run it and see the answer and sources.
    response = rag.query(query)
    
    print(f"ANSWER: {response.answer}")
    print(f"GROUNDED: {response.grounded}")
    print(f"\nSOURCES ({len(response.citations)}):")
    for i, c in enumerate(response.citations[:5]):
        print(f"[{i+1}] {c.source_name} (Relevance: {c.relevance_score:.3f})")
        print(f"    Snippet: {c.snippet[:150]}...")

if __name__ == "__main__":
    queries = [
        "Who is the HOD of CSE department?",
        "Who is the HOD of CSE-DS department?",
        "Who is the HOD of CSE-AIML department?",
        "Who is the HOD of CSE-IOT department?"
    ]
    for q in queries:
        debug_query(q)
