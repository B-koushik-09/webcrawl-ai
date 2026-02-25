"""
Migration script: Patch existing metadata.pkl with 'dept' field (CSE/OTHER).
This avoids a full rebuild — just loads, classifies, and saves back.
"""
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import CSE_DEPT_KEYWORDS

STORAGE_DIR = Path(__file__).parent / "storage"
META_PATH = STORAGE_DIR / "metadata.pkl"

def classify_dept(content, source_name="", section_title=""):
    combined = f"{content} {source_name} {section_title}".lower()
    return 'CSE' if any(kw in combined for kw in CSE_DEPT_KEYWORDS) else 'OTHER'

def migrate():
    if not META_PATH.exists():
        print("[ERROR] metadata.pkl not found. Build index first.")
        return
    
    # Load existing items
    with open(META_PATH, "rb") as f:
        items = pickle.load(f)
    
    print(f"[*] Loaded {len(items)} chunks from metadata.pkl")
    
    # Classify each chunk
    cse_count = 0
    other_count = 0
    for item in items:
        section = item.metadata.get('section', '')
        dept = classify_dept(item.content, item.source_name, section)
        item.metadata['dept'] = dept
        if dept == 'CSE':
            cse_count += 1
        else:
            other_count += 1
    
    # Save back
    with open(META_PATH, "wb") as f:
        pickle.dump(items, f)
    
    print(f"[OK] Migration complete!")
    print(f"     CSE chunks:   {cse_count}")
    print(f"     OTHER chunks: {other_count}")
    print(f"     Total:        {len(items)}")
    
    # Show some CSE examples
    if cse_count > 0:
        print(f"\n--- Sample CSE chunks ---")
        shown = 0
        for item in items:
            if item.metadata.get('dept') == 'CSE' and shown < 5:
                print(f"  [{shown+1}] {item.source_name[:60]} | {item.content[:80]}...")
                shown += 1

if __name__ == "__main__":
    migrate()
