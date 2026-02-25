"""
Extract and display markdown content from each page in scraped_pages.json.
Also applies the repetition-based text cleaner to show cleaned output.
"""

import json
from pathlib import Path

# Import the text cleaner
from modules.text_cleaner import build_noise_lines, clean_markdown

DATA_FILE = Path("data/scraped_pages.json")

def main():
    # Load the scraped pages
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    pages = data.get("pages", [])
    print(f"=" * 70)
    print(f"SCRAPED PAGES ANALYSIS")
    print(f"Total pages: {len(pages)}")
    print(f"=" * 70)
    
    # Step 1: Build noise lines from ALL pages (repetition-based detection)
    print(f"\n[1] Building noise lines from {len(pages)} pages...")
    noise_lines = build_noise_lines(pages)
    print(f"    Found {len(noise_lines)} noise lines (repetitive boilerplate)")
    
    if noise_lines:
        print(f"\n    Sample noise lines:")
        for line in sorted(list(noise_lines)[:5]):
            print(f"      - '{line[:70]}...'")
    
    # Step 2: Display each page's content
    print(f"\n[2] Extracting content from each page:\n")
    print("=" * 70)
    
    for i, page in enumerate(pages, 1):
        url = page.get("url", "N/A")
        title = page.get("title", "N/A")
        markdown = page.get("markdown", "")
        clean_text = page.get("clean_text", "")
        
        # Apply our cleaner
        cleaned_md = clean_markdown(markdown, noise_lines)
        
        print(f"\n--- PAGE {i} ---")
        print(f"URL: {url}")
        print(f"Title: {title}")
        print(f"Markdown length: {len(markdown)} chars")
        print(f"Clean text length: {len(clean_text)} chars")
        print(f"After our cleaner: {len(cleaned_md)} chars")
        
        # Show a preview of cleaned content
        if cleaned_md:
            preview = cleaned_md[:500].replace('\n', ' ')
            print(f"\nPREVIEW (first 500 chars):")
            print(f"  {preview}...")
        else:
            print(f"\nPREVIEW: (empty after cleaning)")
        
        print(f"\n" + "-" * 70)
        
        # Limit output for readability
        if i >= 10:
            print(f"\n... and {len(pages) - 10} more pages")
            break
    
    # Step 3: Summary stats
    print(f"\n" + "=" * 70)
    print(f"SUMMARY STATISTICS")
    print(f"=" * 70)
    
    total_markdown_chars = sum(len(p.get("markdown", "")) for p in pages)
    total_clean_chars = sum(len(clean_markdown(p.get("markdown", ""), noise_lines)) for p in pages)
    
    print(f"Total raw markdown: {total_markdown_chars:,} characters")
    print(f"Total after cleaning: {total_clean_chars:,} characters")
    print(f"Reduction: {100 - (total_clean_chars / total_markdown_chars * 100):.1f}%")


if __name__ == "__main__":
    main()
