"""
Debug script to check what's in the FAISS index and why retrieval is wrong.
"""
import sys
import pickle
from pathlib import Path

# Load the metadata to see what chunks exist
metadata_path = Path("storage/metadata.pkl")

if metadata_path.exists():
    with open(metadata_path, "rb") as f:
        items = pickle.load(f)
    
    print(f"Total indexed items: {len(items)}")
    print()
    
    # Search for chunks containing "1995" or "established"
    print("=" * 70)
    print("CHUNKS CONTAINING '1995' (establishment year)")
    print("=" * 70)
    
    found_1995 = []
    for item in items:
        if "1995" in item.content:
            found_1995.append(item)
    
    print(f"Found {len(found_1995)} chunks with '1995'")
    for item in found_1995[:5]:
        print(f"\n--- Source: {item.source_name} ---")
        print(f"Content preview: {item.content[:300]}...")
    
    print()
    print("=" * 70)
    print("CHUNKS CONTAINING '14 B.Tech' or '13 M.Tech'")
    print("=" * 70)
    
    found_programs = []
    for item in items:
        if "14 B.Tech" in item.content or "13 M.Tech" in item.content or "14 b.tech" in item.content.lower():
            found_programs.append(item)
    
    print(f"Found {len(found_programs)} chunks with program counts")
    for item in found_programs[:3]:
        print(f"\n--- Source: {item.source_name} ---")
        print(f"Content preview: {item.content[:300]}...")
    
    # Test actual retrieval
    print()
    print("=" * 70)
    print("TESTING RETRIEVAL: 'When was VNRVJIET established?'")
    print("=" * 70)
    
    from modules.embeddings import KnowledgeIndex
    
    index = KnowledgeIndex()
    index.load_index()
    
    results = index.search("When was VNRVJIET established?", top_k=5, min_similarity=0.3)
    
    print(f"\nTop 5 retrieved chunks:")
    for i, (item, score) in enumerate(results, 1):
        print(f"\n[{i}] Score: {score:.3f} | Source: {item.source_name}")
        print(f"    Content: {item.content[:200]}...")

else:
    print("No metadata.pkl found!")
