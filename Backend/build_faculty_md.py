"""
Build Faculty MD - Separate PDF Processing Script

This script processes downloaded PDFs AFTER indexing to build faculty.md.
Run this separately from the indexing pipeline for stability.

Usage:
    python build_faculty_md.py             # Process all PDFs
    python build_faculty_md.py --reindex   # Also re-index with faculty data

This separation ensures:
- Fast, stable indexing (crawl + download only)
- Safe PDF parsing (no crashes during indexing)
- Better architectural design
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, PDF_DIR
from modules.embeddings import KnowledgeIndex

# Import the existing faculty extractor
from extract_faculty import main as extract_faculty_main


def build_faculty_md(delete_pdfs: bool = False):
    """
    Step 1: Extract faculty data from PDFs → faculty.md
    """
    print("=" * 60)
    print("STEP 1: Extracting Faculty Data from PDFs")
    print("=" * 60)
    
    extract_faculty_main(delete_pdfs=delete_pdfs)
    
    faculty_file = DATA_DIR / 'faculty.md'
    if faculty_file.exists():
        print(f"\n[OK] faculty.md created at: {faculty_file}")
        return True
    else:
        print("\n[WARN] faculty.md was not created (no faculty PDFs found?)")
        return False


def reindex_with_faculty():
    """
    Step 2: Add faculty.md to the existing index
    """
    print("\n" + "=" * 60)
    print("STEP 2: Adding Faculty Data to Index")
    print("=" * 60)
    
    faculty_file = DATA_DIR / 'faculty.md'
    if not faculty_file.exists():
        print("[ERROR] faculty.md not found. Run step 1 first.")
        return False
    
    # Load existing index
    index = KnowledgeIndex()
    
    if index.index is None or index.index.ntotal == 0:
        print("[WARN] No existing index found. Run full indexing first.")
        return False
    
    print(f"[INFO] Existing index has {index.index.ntotal} items")
    
    # Add faculty data
    print("[*] Adding faculty.md to index...")
    added = index.add_markdown_file(faculty_file, "Faculty Directory")
    
    if added > 0:
        # Save updated index
        index.save_index()
        print(f"[OK] Added {added} faculty chunks to index")
        print(f"[OK] Index now has {index.index.ntotal} total items")
        return True
    else:
        print("[WARN] No faculty chunks added")
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build faculty.md from PDFs")
    parser.add_argument('--reindex', action='store_true', 
                       help='Also add faculty.md to existing index')
    parser.add_argument('--delete-pdfs', action='store_true',
                       help='Delete faculty PDFs after extraction')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Build Faculty MD - Post-Indexing PDF Processor")
    print("=" * 60)
    print(f"PDF Directory: {PDF_DIR}")
    print(f"Output: {DATA_DIR / 'faculty.md'}")
    print("=" * 60 + "\n")
    
    # Step 1: Extract faculty data
    success = build_faculty_md(delete_pdfs=args.delete_pdfs)
    
    # Step 2: Optionally add to index
    if success and args.reindex:
        reindex_with_faculty()
    elif success:
        print("\n[TIP] Run with --reindex to add faculty.md to the search index")
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
