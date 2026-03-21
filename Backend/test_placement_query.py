import sys
import os
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.append('.')

from modules.rag_engine import get_rag_engine
from modules.embeddings import KnowledgeIndex

def test_query():
    print("Loading Index...")
    idx = KnowledgeIndex()
    
    print("Initializing RAG Engine...")
    rag = get_rag_engine()
    rag.set_index(idx)
    
    query = "cse training and placement department faculty list"
    print(f"\nQuerying: {query}")
    res = rag.query(query)
    
    with open('placement_test_out.txt', 'w', encoding='utf-8') as f:
        f.write(res.answer)
    print("Done. Wrote to placement_test_out.txt")

if __name__ == "__main__":
    test_query()
