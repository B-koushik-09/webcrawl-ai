import os
import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime

# Add Backend to path
sys.path.append(str(Path(__file__).parent.parent))

from modules.embeddings import KnowledgeIndex, KnowledgeItem

def batch_update_index(filenames):
    """Update multiple files in the index efficiently."""
    print(f"[*] Loading knowledge index...")
    index = KnowledgeIndex()
    if not index.load_index():
        print("[ERROR] Failed to load index.")
        return
    
    initial_count = len(index.items)
    print(f"[OK] Loaded {initial_count} items from index.")
    
    modified_sources = set()
    for filename in filenames:
        if filename.startswith('PDF_'):
            source_name = filename.replace('.md', '.pdf').replace('PDF_', '')
        else:
            # Try to get title from file
            file_path = Path(r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages") / filename
            title = "VNRVJIET"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('# '):
                            title = line[2:].strip()
                            break
            source_name = title
        modified_sources.add(source_name)
    
    print(f"[*] Sources to refresh: {modified_sources}")
    
    # Filter out old items
    new_items = []
    indices_to_keep = []
    removed_count = 0
    
    for i, item in enumerate(index.items):
        if item.source_name in modified_sources:
            removed_count += 1
        else:
            indices_to_keep.append(i)
            new_items.append(item)
    
    print(f"[*] Removing {removed_count} old chunks...")
    
    if removed_count > 0:
        # Reconstruct FAISS index
        if index.index and indices_to_keep:
            print(f"[*] Reconstructing {len(indices_to_keep)} vectors...")
            # Use reconstruct_n for efficiency if possible, or loop if needed
            # For FlatL2, we can just get all vectors
            all_vectors = index.index.reconstruct_n(0, index.index.ntotal)
            filtered_vectors = all_vectors[indices_to_keep]
            
            import faiss
            new_faiss_index = faiss.IndexFlatL2(index.dimension)
            new_faiss_index.add(filtered_vectors.astype('float32'))
            index.index = new_faiss_index
            print(f"[*] FAISS index rebuilt with {index.index.ntotal} vectors")
        else:
            # If everything was removed or index empty
            import faiss
            index.index = faiss.IndexFlatL2(index.dimension)
            
        index.items = new_items
        # Re-assign IDs to be contiguous
        for i, item in enumerate(index.items):
            item.id = i
    
    print(f"[OK] Index cleaned. Now re-indexing {len(filenames)} files...")
    
    # Re-index modified files
    for filename in filenames:
        file_path = Path(r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages") / filename
        if not file_path.exists():
            print(f"[SKIP] {filename} not found")
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if filename.startswith('PDF_'):
            source_name = filename.replace('.md', '.pdf').replace('PDF_', '')
            pdf_inputs = [{
                'content': content,
                'pdf_name': source_name.replace('.pdf', ''),
                'pdf_path': source_name,
                'doc_type': 'pdf',
                'page_number': 1,
                'metadata': {'source': 'batch_update'}
            }]
            index.add_pdf_chunks(pdf_inputs)
        elif filename == 'faculty.md':
            index.add_markdown_file(file_path, "Faculty Directory")
        else:
            # Webpage
            title = "VNRVJIET"
            url = "https://vnrvjiet.ac.in/"
            for line in content.split('\n')[:10]:
                if line.startswith('# '): title = line[2:].strip()
                if 'Source:' in line and '(' in line:
                    url = line.split('(')[1].split(')')[0]
            
            pages = [{
                'title': title,
                'url': url,
                'markdown': content,
                'scraped_at': datetime.now().isoformat()
            }]
            index.add_webpage_content(pages, skip_cleaning=True)

    print(f"[*] Saving updated index...")
    index.save_index()
    index.save_status(True, index.status.get('url', ''), 0, 0, len(index.items))
    print(f"[DONE] Index updated. Final count: {len(index.items)} chunks")

if __name__ == "__main__":
    # The same list as in improve_headings_targeted.py
    TARGET_FILES = [
        "PDF_CSE_R22_I_Year.md", "PDF_CSE_R22_II_Year.md", "PDF_CSE_R22_III_Year.md", "PDF_CSE_R22_IV_Year.md",
        "060_cse.md", "129_cse-aiml-and-iot.md", "038_cse-ds-and-cys.md",
        "faculty.md",
        "074_hostel.md", "PDF_VNRVJIET_Hostel_Brochure_2025_26.md", "PDF_Application_for_Admission_in_the_Hostel_of__VNRVJIET.md"
    ]
    batch_update_index(TARGET_FILES)
