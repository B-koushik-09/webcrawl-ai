import sys
import os
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.append('.')

from modules.rag_engine import get_rag_engine
from modules.embeddings import KnowledgeIndex

def test_hod():
    # Redirect stdout to a file
    sys.stdout = open('hod_diagnostic.txt', 'w', encoding='utf-8')
    sys.stderr = sys.stdout
    
    print("Loading Index...")
    idx = KnowledgeIndex()
    
    print("Initializing RAG Engine...")
    rag = get_rag_engine()
    rag.set_index(idx)
    
    queries = [
        "Who is the HOD of CSE?",
        "Who is the HOD of CSE-DS?",
        "Who is the HOD of CSE-AIML?",
        "Who is the HOD of CSE-IOT?",
        "Who is the HOD of EEE?",
    ]
    
    for query in queries:
        print(f"\n{'='*80}")
        print(f"TESTING QUERY: {query}")
        print(f"{'='*80}")
        res = rag.query(query)
        print(f"\nFINAL ANSWER:\n{res.answer}\n")
        print(f"CONFIDENCE: {res.confidence}")
        
    print("Done testing.")

if __name__ == "__main__":
    test_hod()
