import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.embeddings import KnowledgeIndex
from modules.rag_engine import RAGEngine

print("[*] Loading existing ChromaDB index...")
index = KnowledgeIndex()
count = index.collection.count()
print(f"[OK] ChromaDB has {count} chunks loaded")

engine = RAGEngine(index)

queries = [
    "Who is the principal of VNRVJIET?",
    "Who is the HOD of CSE department?",
    "What are the fees for B.Tech?",
]

def safe_print(text):
    print(text.encode('ascii', 'ignore').decode('ascii'))

for q in queries:
    safe_print(f"\n{'='*60}")
    safe_print(f"Q: {q}")
    safe_print(f"{'='*60}")
    res = engine.query(q)
    safe_print(f"\nA: {res.answer}")
    print(f"Confidence: {res.confidence}")
    if res.citations:
        print(f"Sources: {[c.source_name for c in res.citations[:3]]}")
