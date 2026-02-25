"""
Quick debug script to check index content.
"""
import pickle
from pathlib import Path

STORAGE_DIR = Path(__file__).parent / "storage"

# Load metadata
with open(STORAGE_DIR / "metadata.pkl", "rb") as f:
    items = pickle.load(f)

print(f"Total items: {len(items)}")
print("\n" + "="*60)
print("ITEMS CONTAINING '1995' (establishment year)")
print("="*60)

found = 0
for item in items:
    if "1995" in item.content:
        found += 1
        print(f"\n[{found}] Source: {item.source_name}")
        print(f"    URL: {item.source_url[:60]}..." if len(item.source_url) > 60 else f"    URL: {item.source_url}")
        # Show context around 1995
        idx = item.content.find("1995")
        start = max(0, idx - 100)
        end = min(len(item.content), idx + 150)
        context = item.content[start:end].replace('\n', ' ')
        print(f"    Context: ...{context}...")
        
        if found >= 10:
            break

print(f"\nTotal chunks with '1995': {found}")
