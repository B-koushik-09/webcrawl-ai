import sys
from pathlib import Path
import pickle

# Add Backend to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.embeddings import KnowledgeIndex

def verify():
    index = KnowledgeIndex()
    if not index.load_index():
        print("Failed to load index")
        return
    
    source_counts = {}
    for item in index.items:
        source_counts[item.source_name] = source_counts.get(item.source_name, 0) + 1
    
    # Check specific targets
    targets = ["CSE R22 III Year", "Faculty Directory", "Hostel"]
    print("-" * 40)
    for t in targets:
        count = source_counts.get(t, 0)
        print(f"{t}: {count} chunks")
    print("-" * 40)
    print(f"Total chunks: {len(index.items)}")

if __name__ == "__main__":
    verify()
