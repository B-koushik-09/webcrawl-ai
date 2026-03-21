import sys
import os
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.append('.')

from modules.rag_engine import get_rag_engine
from modules.embeddings import KnowledgeIndex

def run_queries():
    print("Loading Index...")
    idx = KnowledgeIndex()
    rag = get_rag_engine()
    rag.set_index(idx)
    
    queries = [
        "What is the highest placement package in CSE?",
        "How many students were placed in Amazon?",
        "cse training and placement department faculty list"
    ]
    
    with open('placement_multi_test.txt', 'w', encoding='utf-8') as f:
        for q in queries:
            print(f"\nQuerying: {q}")
            res = rag.query(q)
            f.write(f"Q: {q}\n")
            f.write(f"Type: {rag._detect_question_type(q)}\n")
            f.write(f"A: {res.answer}\n")
            f.write("-" * 40 + "\n")
            
    print("Done. Check placement_multi_test.txt")

if __name__ == "__main__":
    run_queries()
