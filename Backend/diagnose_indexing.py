
import json
import os
import re
from pathlib import Path

# Need to adjust paths since running from project root presumably
# But I am writing directly to absolute path, so paths in script should be absolute too or relative to execution context.
# Assuming running from Backend dir.

DATA_DIR = Path(r'c:\VNRVJIET\Projects\ai-chat\Backend\data')
CLEANED_DIR = Path(r'c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages')

def diagnose():
    print("=== Indexing Diagnosis ===")
    
    # 1. Check scraped_pages.json
    scraped_file = DATA_DIR / 'scraped_pages.json'
    if not scraped_file.exists():
        print(f"ERROR: scraped_pages.json not found at {scraped_file}")
    else:
        try:
            with open(scraped_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            

            # Write to file
            with open('diagnosis_result.txt', 'w', encoding='utf-8') as out:
                out.write(f"scraped_pages.json 'pages' count: {len(pages)}\n")
                out.write(f"scraped_pages.json 'pages_count' header: {data.get('pages_count')}\n")
                
                # Analyze scraped pages content length
                short_pages = 0
                duplicate_urls = set()
                seen_urls = set()
                valid_pages = 0
                
                for p in pages:
                    content = p.get('clean_text', '') or p.get('markdown', '')
                    if len(content) < 50:
                        short_pages += 1
                    else:
                        valid_pages += 1
                    
                    url = p.get('url')
                    if url in seen_urls:
                        duplicate_urls.add(url)
                    seen_urls.add(url)
                    
                out.write(f"Pages with content < 50 chars: {short_pages}\n")
                out.write(f"Duplicate URLs in json: {len(duplicate_urls)}\n")
                out.write(f"Unique URLs in json: {len(seen_urls)}\n")
                out.write(f"Expected indexed from JSON (valid pages): {valid_pages}\n")
        
        except Exception as e:
            with open('diagnosis_result.txt', 'w', encoding='utf-8') as out:
                out.write(f"Error reading JSON: {e}\n")

    # 2. Check cleaned_pages directory
    if not CLEANED_DIR.exists():
        with open('diagnosis_result.txt', 'a', encoding='utf-8') as out:
            out.write(f"ERROR: cleaned_pages directory not found at {CLEANED_DIR}\n")
    else:
        md_files = list(CLEANED_DIR.glob('*.md'))
        with open('diagnosis_result.txt', 'a', encoding='utf-8') as out:
            out.write(f"\ncleaned_pages/*.md count: {len(md_files)}\n")
        
        # Analyze cleaned pages
        valid_md_files = 0
        short_md_files = 0
        for md_file in md_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                if len(content) >= 50:
                    valid_md_files += 1
                else:
                    short_md_files += 1
            except Exception as e:
                with open('diagnosis_result.txt', 'a', encoding='utf-8') as out:
                    out.write(f"Error reading {md_file}: {e}\n")
        
        with open('diagnosis_result.txt', 'a', encoding='utf-8') as out:
            out.write(f"Valid MD files (content >= 50 chars): {valid_md_files}\n")
            out.write(f"Short MD files (< 50 chars): {short_md_files}\n")
    
    # 3. List
    if 'pages' in locals() and 'md_files' in locals():
        with open('diagnosis_result.txt', 'a', encoding='utf-8') as out:
            out.write("\nComparison Check:\n")
            out.write(f"JSON Valid Pages: {valid_pages}\n")
            out.write(f"MD  Valid Files: {valid_md_files}\n")
            diff = valid_md_files - valid_pages
            out.write(f"Discripancy: {diff}\n")

if __name__ == "__main__":
    diagnose()
