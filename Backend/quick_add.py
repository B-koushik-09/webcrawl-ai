"""
Quick Add to Index
==================
Appends a single file to the existing index WITHOUT full rebuild.

Usage:
    python quick_add.py faq.md

Note: 
- This ADDS chunks, doesn't remove old ones
- For clean index, do full rebuild periodically
- Fast: ~30 seconds instead of 12 minutes
"""

import sys
from pathlib import Path

def quick_add(filename: str):
    """Add/update a single file to the existing index."""
    print(f"\n{'='*60}")
    print(f"QUICK ADD TO INDEX: {filename}")
    print(f"{'='*60}\n")
    
    # Setup paths
    base_dir = Path(__file__).parent
    cleaned_dir = base_dir / 'cleaned_pages'
    target_file = cleaned_dir / filename
    
    if not target_file.exists():
        # Try without cleaned_pages prefix
        target_file = base_dir / filename
        if not target_file.exists():
            print(f"[ERROR] File not found: {filename}")
            return False
    
    print(f"[*] Loading existing index...")
    
    from modules.embeddings import KnowledgeIndex
    index = KnowledgeIndex()
    
    if not index.load_index():
        print(f"[ERROR] No existing index. Run 'Rebuild Index' first.")
        return False
    
    initial_count = index.index.ntotal if index.index else 0
    print(f"[OK] Index loaded: {initial_count} chunks")
    
    # Read file content
    print(f"[*] Reading {target_file.name}...")
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if len(content) < 50:
        print(f"[ERROR] File too short (< 50 chars)")
        return False
    
    # Extract title from markdown
    title = target_file.stem.replace('_', ' ').title()
    lines = content.split('\n')
    for line in lines[:10]:
        if line.startswith('# '):
            title = line[2:].strip()
            break
    
    print(f"[*] Title: {title}")
    
    # Add as webpage content
    pages = [{
        'title': title,
        'url': f"custom/{target_file.name}",
        'markdown': content,
        'scraped_at': ''
    }]
    
    print(f"[*] Adding to index (skip_cleaning=True)...")
    chunks_added = index.add_webpage_content(pages, skip_cleaning=True)
    
    # Save updated index
    print(f"[*] Saving index...")
    index.save_index()
    
    final_count = index.index.ntotal if index.index else 0
    
    print(f"\n{'='*60}")
    print(f"✅ SUCCESS!")
    print(f"   File: {filename}")
    print(f"   Chunks added: {final_count - initial_count}")
    print(f"   Total chunks: {final_count}")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_add.py <filename>")
        print("Example: python quick_add.py faq.md")
        sys.exit(1)
    
    filename = sys.argv[1]
    success = quick_add(filename)
    sys.exit(0 if success else 1)
