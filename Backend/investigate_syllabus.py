"""
Investigate: Are R22/R25 syllabus PDFs indexed in FAISS?
Searches metadata.pkl for syllabus-related chunks.
"""
import pickle
from pathlib import Path
from collections import defaultdict

STORAGE_DIR = Path(__file__).parent / "storage"

with open(STORAGE_DIR / "metadata.pkl", "rb") as f:
    items = pickle.load(f)

print(f"Total chunks in index: {len(items)}\n")

# =============================================
# 1. Find all chunks with "R22" or "R25" in source name or content
# =============================================
r22_chunks = []
r25_chunks = []
syllabus_sources = defaultdict(list)  # source_name -> [chunks]

for item in items:
    combined = f"{item.source_name} {item.content}".lower()
    source_lower = item.source_name.lower()
    
    is_syllabus = 'syllabus' in combined or 'r22' in combined or 'r25' in combined
    
    if not is_syllabus:
        continue
    
    if 'r22' in combined:
        r22_chunks.append(item)
    if 'r25' in combined:
        r25_chunks.append(item)
    
    syllabus_sources[item.source_name].append(item)

print("=" * 70)
print("SYLLABUS CHUNKS SUMMARY")
print("=" * 70)
print(f"R22-related chunks: {len(r22_chunks)}")
print(f"R25-related chunks: {len(r25_chunks)}")
print(f"Unique sources with syllabus content: {len(syllabus_sources)}")

# =============================================
# 2. Show source breakdown
# =============================================
print(f"\n{'=' * 70}")
print("SOURCE BREAKDOWN (syllabus-related)")
print("=" * 70)
for source, chunks in sorted(syllabus_sources.items(), key=lambda x: -len(x[1])):
    dept = chunks[0].metadata.get('dept', '?')
    src_type = chunks[0].source_type
    print(f"\n[{src_type.upper()}] [{dept}] {source}")
    print(f"  Chunks: {len(chunks)}")
    # Show first chunk snippet
    print(f"  Sample: {chunks[0].content[:120].replace(chr(10), ' ')}...")

# =============================================
# 3. Specifically check: are CSE R22 year-wise syllabus PDFs indexed?
# =============================================
print(f"\n{'=' * 70}")
print("CSE R22 YEAR-WISE CHECK")
print("=" * 70)
for year_kw in ['i year', '1st year', 'first year', 'ii year', '2nd year', 'second year', 
                'iii year', '3rd year', 'third year', 'iv year', '4th year', 'fourth year']:
    found = [c for c in r22_chunks if year_kw in c.content.lower() and 'cse' in f"{c.source_name} {c.content}".lower()]
    if found:
        print(f"\n  '{year_kw}' + CSE + R22: {len(found)} chunks")
        print(f"    Source: {found[0].source_name[:60]}")
        print(f"    Type: {found[0].source_type}")
        print(f"    Snippet: {found[0].content[:100].replace(chr(10), ' ')}...")

# =============================================
# 4. Check for specific subjects (machine learning, data structures, etc.)
# =============================================
print(f"\n{'=' * 70}")
print("SUBJECT-LEVEL CHECK IN CSE CHUNKS")
print("=" * 70)
subjects = ['machine learning', 'data structures', 'operating systems', 'dbms', 
            'computer networks', 'compiler', 'software engineering', 'algorithms']
for subj in subjects:
    found = [c for c in items if subj in c.content.lower() and c.metadata.get('dept') == 'CSE']
    pdf_found = [c for c in found if c.source_type == 'pdf']
    web_found = [c for c in found if c.source_type != 'pdf']
    print(f"  '{subj}': {len(found)} total ({len(pdf_found)} PDF, {len(web_found)} web)")

# =============================================
# 5. Check for R25
# =============================================
print(f"\n{'=' * 70}")
print("R25 CHECK")
print("=" * 70)
if r25_chunks:
    for chunk in r25_chunks[:5]:
        print(f"\n  Source: {chunk.source_name[:60]}")
        print(f"  Type: {chunk.source_type} | Dept: {chunk.metadata.get('dept', '?')}")
        print(f"  Content: {chunk.content[:150].replace(chr(10), ' ')}...")
else:
    print("  NO R25 chunks found in index!")
