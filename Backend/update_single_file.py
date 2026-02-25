"""
Incremental Index Update Script
================================
Re-index a single file without rebuilding the entire index.

Usage:
    python update_single_file.py 134_admission.md

This is MUCH faster than full rebuild for small edits!
"""

import sys
import json
from pathlib import Path
from modules.embeddings import KnowledgeIndex

def update_single_file(filename: str):
    """
    Update a single file in the existing index.
    
    Args:
        filename: Name of the file in cleaned_pages/ (e.g., '134_admission.md')
    """
    print(f"\n{'='*60}")
    print(f"INCREMENTAL INDEX UPDATE")
    print(f"{'='*60}\n")
    
    # Initialize paths
    base_dir = Path(__file__).parent
    cleaned_pages_dir = base_dir / 'cleaned_pages'
    target_file = cleaned_pages_dir / filename
    
    if not target_file.exists():
        print(f"[ERROR] File not found: {target_file}")
        return False
    
    print(f"[*] Loading existing index...")
    index = KnowledgeIndex()
    
    if not index.load_index():
        print(f"[ERROR] No existing index found. Run 'Rebuild Index' first.")
        return False
    
    initial_count = index.index.ntotal if index.index else 0
    print(f"[OK] Index loaded: {initial_count} chunks")
    
    # Step 1: Find and remove old chunks from this file
    print(f"\n[*] Removing old chunks from '{filename}'...")
    
    # Determine the source name that was used during indexing
    # For 134_admission.md (a webpage), source_name is typically "VNRVJIET"
    # For PDF_*.md files, source_name is the PDF name
    
    if filename.startswith('PDF_'):
        # PDF file: source name is the stem without 'PDF_' prefix
        source_name = filename.replace('.md', '.pdf').replace('PDF_', '')
    else:
        # Webpage file: source name is typically "VNRVJIET"
        # We need to extract the title from the file itself
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to extract title from markdown
        lines = content.split('\n')
        title = "VNRVJIET"  # default
        for line in lines:
            if line.strip().startswith('# '):
                title = line.strip()[2:].strip()
                break
            elif line.strip().startswith('**Source:**'):
                # Format: **Source:** [title](url)
                import re
                match = re.search(r'\*\*Source:\*\*\s*\[([^\]]+)\]', line)
                if match:
                    title = match.group(1)
                    break
        
        source_name = title
    
    print(f"[*] Looking for chunks with source_name: '{source_name}'")
    
    # Remove old chunks
    chunks_to_keep = []
    chunks_removed = 0
    
    if hasattr(index, 'metadata') and index.metadata:
        for i, meta in enumerate(index.metadata):
            if meta.source_name != source_name:
                chunks_to_keep.append(i)
            else:
                chunks_removed += 1
    
    print(f"[OK] Found {chunks_removed} old chunks to remove")
    
    if chunks_removed > 0:
        # Rebuild index vectors without the removed chunks
        import numpy as np
        
        if index.index and len(chunks_to_keep) > 0:
            # Extract remaining vectors
            old_vectors = np.array([index.index.reconstruct(i) for i in chunks_to_keep])
            
            # Clear and rebuild
            index.index.reset()
            index.index.add(old_vectors)
            index.metadata = [index.metadata[i] for i in chunks_to_keep]
            
            print(f"[OK] Removed {chunks_removed} chunks. Index now has {index.index.ntotal} chunks")
        else:
            print(f"[WARN] No chunks to keep or index is empty")
    
    # Step 2: Re-add updated chunks from the file
    print(f"\n[*] Re-indexing '{filename}'...")
    
    # Read the updated file
    with open(target_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Determine file type and index accordingly
    if filename.startswith('PDF_'):
        # PDF file
        pdf_inputs = [{
            'content': content,
            'pdf_name': source_name.replace('.pdf', ''),
            'pdf_path': source_name,
            'doc_type': 'pdf',
            'page_number': 1,
            'metadata': {'source': 'incremental_update'}
        }]
        
        chunks_added = index.add_pdf_chunks(pdf_inputs)
    elif filename == 'faculty.md':
        # Faculty markdown
        chunks_added = index.add_markdown_file(target_file, "Faculty Directory")
    else:
        # Webpage markdown
        # Extract title and URL from content
        title = source_name
        url = "https://vnrvjiet.ac.in/"
        
        for line in content.split('\n')[:20]:
            if '**Source:**' in line:
                import re
                url_match = re.search(r'\(([^)]+)\)', line)
                if url_match:
                    url = url_match.group(1)
                break
        
        pages = [{
            'title': title,
            'url': url,
            'markdown': content,
            'scraped_at': ''
        }]
        
        chunks_added = index.add_webpage_content(pages, skip_cleaning=True)
    
    print(f"[OK] Added {chunks_added} new chunks")
    
    # Step 3: Save the updated index
    print(f"\n[*] Saving updated index...")
    index.save_index()
    
    final_count = index.index.ntotal if index.index else 0
    print(f"[OK] Index saved: {final_count} total chunks")
    
    # Update status
    # Load current status
    status_file = base_dir / 'storage' / 'index_status.json'
    if status_file.exists():
        with open(status_file, 'r') as f:
            status = json.load(f)
        
        status['chunks'] = final_count
        status['last_updated'] = __import__('datetime').datetime.now().isoformat()
        
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ SUCCESS! Updated '{filename}' in {final_count} chunks")
    print(f"   Removed: {chunks_removed} old chunks")
    print(f"   Added: {chunks_added} new chunks")
    print(f"   Total: {final_count} chunks")
    print(f"{'='*60}\n")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_single_file.py <filename>")
        print("Example: python update_single_file.py 134_admission.md")
        sys.exit(1)
    
    filename = sys.argv[1]
    success = update_single_file(filename)
    
    sys.exit(0 if success else 1)
