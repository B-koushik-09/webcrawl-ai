import faiss
import sys
from pathlib import Path

index_path = Path(r"c:\VNRVJIET\Projects\ai-chat\Backend\storage\knowledge.faiss")
if not index_path.exists():
    print("Index not found")
    sys.exit(1)

index = faiss.read_index(str(index_path))
print(f"Index type: {type(index)}")
print(f"Total items: {index.ntotal}")

try:
    vec = index.reconstruct(0)
    print("Reconstruction successful for index 0")
    print(f"Vector shape: {vec.shape}")
except Exception as e:
    print(f"Reconstruction failed: {e}")
