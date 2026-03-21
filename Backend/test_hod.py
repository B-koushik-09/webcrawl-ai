import sys
import os
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.append('.')

from modules.rag_engine import get_rag_engine
from modules.embeddings import KnowledgeIndex

def test_hod():
    print("Loading Index...")
    idx = KnowledgeIndex()
    
    print("Initializing RAG Engine...")
    rag = get_rag_engine()
    rag.set_index(idx)
    
    query = "Who is the HOD of EEE?"
    print(f"\nQuerying: {query}")
    res = rag.query(query)
    
    with open('hod_result.txt', 'w', encoding='utf-8') as f:
        f.write(res.answer)
    print("Done. Wrote to hod_result.txt")

if __name__ == "__main__":
    test_hod()
