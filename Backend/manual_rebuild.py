import asyncio
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.embeddings import KnowledgeIndex
from modules.rag_engine import RAGEngine
from config import BASE_DIR

async def main():
    print("[*] Starting manual index rebuild...")
    index = KnowledgeIndex()
    index.clear()
    
    cleaned_dir = BASE_DIR / 'cleaned_pages'
    all_md_files = list(cleaned_dir.glob('*.md'))
    print(f"[*] Found {len(all_md_files)} files.")
    
    import re
    webpage_pages = []
    pdf_inputs = []
    custom_pages = []
    
    for f in all_md_files:
        if f.name.startswith('PDF_'):
            try:
                content = f.read_text('utf-8', errors='replace')
                if len(content) < 50: continue
                title = f.stem[4:]
                pdf_inputs.append({
                    'content': content,
                    'pdf_name': title,
                    'pdf_path': str(f.name).replace('.md', '.pdf'),
                    'doc_type': 'pdf',
                    'page_number': 1
                })
            except Exception as e: print(e)
        elif re.match(r'^\d{3}_', f.name):
            try:
                content = f.read_text('utf-8', errors='replace')
                if len(content) < 50: continue
                title = "VNRVJIET"
                lines = content.split('\n')
                for line in lines[:15]:
                    if line.startswith('# '):
                        title = line[2:].strip()
                        break
                webpage_pages.append({
                    'title': title,
                    'url': f"https://vnrvjiet.ac.in/{f.name}",
                    'markdown': content
                })
            except Exception as e: print(e)
        else:
            try:
                content = f.read_text('utf-8', errors='replace')
                if len(content) < 50: continue
                title = f.stem.replace('_', ' ').title()
                custom_pages.append({
                    'title': title,
                    'url': f"custom/{f.name}",
                    'markdown': content
                })
            except Exception as e: print(e)
            
    print(f"[*] Batching: {len(webpage_pages)} web, {len(custom_pages)} custom, {len(pdf_inputs)} pdfs")
    
    if webpage_pages:
        index.add_webpage_content(webpage_pages, skip_cleaning=True)
    if custom_pages:
        index.add_webpage_content(custom_pages, skip_cleaning=True)
    if pdf_inputs:
        index.add_pdf_chunks(pdf_inputs)
        
    print("[*] Testing RAG Engine")
    engine = RAGEngine(index)
    
    queries = [
        "Who is the principal of VNRVJIET?",
        "Who is the HOD of CSE department?",
        "What are the fees for B.Tech?"
    ]
    
    for q in queries:
        print(f"\nQ: {q}")
        res = engine.query(q)
        print(f"A: {res.answer}")

    # --- Count Verification Test (consolidated from verify_count.py) ---
    print("\n" + "="*50)
    print("[*] Running Final Count Verification Test...")
    count_query = "How many programmes does VNRVJIET offer?"
    print(f"Query: '{count_query}'")
    
    response = engine.query(count_query)
    print("\n=== Answer ===")
    print(response.answer)
    
    found_summary = False
    for i, cit in enumerate(response.citations[:5]):
        # Updated to check for 17 B.Tech and 15 M.Tech as per latest config
        if "17 B.Tech" in cit.snippet and "15 M.Tech" in cit.snippet:
            print(f"\n[Chunk {i+1}] >>> SUCCESS: Found the summary chunk in {cit.source_name}")
            found_summary = True
            break
            
    if found_summary:
        print("\n[OK] TEST PASSED: Program count summary chunk retrieved.")
    else:
        print("\n[!] TEST FAILED: Summary chunk NOT found in top 5 results.")
    print("="*50 + "\n")
        
if __name__ == '__main__':
    asyncio.run(main())
