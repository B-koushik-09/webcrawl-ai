import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from modules.rag_engine import get_rag_engine
from modules.embeddings import get_knowledge_index

print("Loading index...")
index = get_knowledge_index()
index.load_status()
engine = get_rag_engine()
engine.set_index(index)

print("\nTesting Query: who is the principal of the college?")

# Let's see what the index actually retrieves before verification
query = "who is the principal of the college?"
print(f"\n--- Raw Retrieval from ChromaDB for '{query}' ---")

# Apply the same query expansion as rag_engine
expanded_query = query
for rule_pattern, keywords in engine.leadership_keywords.items():
    import re
    if re.search(rule_pattern, query, re.IGNORECASE):
        expanded_query = f"{query} {keywords}"
        break
        
print(f"Expanded query: {expanded_query}")

results = index.search(expanded_query, top_k=5, min_similarity=0.3)
for i, (item, score) in enumerate(results):
    print(f"\nResult {i+1} (Score: {score:.3f}):")
    print(f"Source: {item.source_name} ({item.doc_type})")
    print(f"Content snippet: {item.content[:200]}...")

print("\n--- Full RAG Engine Execution ---")
response = engine.query(query)
print("\nFinal Answer:", response.answer)
print("Confidence:", response.confidence)
print("Grounded:", response.grounded)
print("Citations:", [c.source_name for c in response.citations])
