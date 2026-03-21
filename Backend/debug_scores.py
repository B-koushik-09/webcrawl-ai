import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from modules.rag_engine import get_rag_engine
from modules.embeddings import get_knowledge_index

index = get_knowledge_index()
index.load_status()
engine = get_rag_engine()
engine.set_index(index)

query = "who is the principal of the college"

# First, see EXACTLY what index.search() returns
print("--- Raw Search Results ---")
# Apply exact same expansion as rag_engine.py
expanded_query = query
for rule_pattern, keywords in engine.leadership_keywords.items():
    import re
    if re.search(rule_pattern, query, re.IGNORECASE):
        expanded_query = f"{query} {keywords}"
        print(f"Expanded to: {expanded_query}")
        break

raw_results = index.search(expanded_query, top_k=5, min_similarity=0.3)
for i, (item, score) in enumerate(raw_results):
    print(f"\nResult {i+1} (Raw Score: {score:.3f}):")
    print(f"Source: {item.source_name} ({item.doc_type})")
    print(f"Content: {item.content[:200]}...")

# Then, see what get_relevant_context() does to those results (boosting/penalties)
print("\n--- Processed RAG Context Results ---")
context_results = engine.get_relevant_context(query)
for i, (item, score) in enumerate(context_results):
    print(f"\nResult {i+1} (Final Score: {score:.3f}):")
    print(f"Source: {item.source_name} ({item.doc_type})")
    print(f"Content: {item.content[:200]}...")

