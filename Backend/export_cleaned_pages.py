"""
Extract and export cleaned markdown content from each page in scraped_pages.json.
Creates separate files for each page in the cleaned_pages/ directory.
"""

import json
import re
from pathlib import Path

# Import the text cleaner
from modules.text_cleaner import build_noise_lines, clean_markdown

DATA_FILE = Path("data/scraped_pages.json")
OUTPUT_DIR = Path("cleaned_pages")

def sanitize_filename(title: str, url: str, index: int) -> str:
    """Create a safe filename from title or URL."""
    # Try to get a meaningful name from title
    if title and title != "VNRVJIET" and title != "404 Page Not Found":
        name = title
    elif url:
        # Extract path from URL
        name = url.split("/")[-1] or url.split("/")[-2] or f"page_{index}"
    else:
        name = f"page_{index:03d}"
    
    # Sanitize: remove special chars, limit length
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = re.sub(r'\s+', '_', name)
    name = name[:60]  # Limit length
    
    return f"{index:03d}_{name}.md"


def extract_page_url_from_content(markdown: str) -> str:
    """Try to extract the page URL from navigation links in the markdown."""
    # Look for patterns like "[Notifications](https://vnrvjiet.ac.in/page-name/#)"
    match = re.search(r'\[Notifications\]\((https://vnrvjiet\.ac\.in/[^#\)]+)', markdown)
    if match:
        return match.group(1)
    return ""


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Load the scraped pages
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    pages = data.get("pages", [])
    print(f"=" * 70)
    print(f"EXPORTING CLEANED PAGES")
    print(f"Total pages: {len(pages)}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print(f"=" * 70)
    
    # Build noise lines from ALL pages
    print(f"\n[1] Building noise lines from {len(pages)} pages...")
    noise_lines = build_noise_lines(pages)
    print(f"    Found {len(noise_lines)} noise lines (repetitive boilerplate)")
    
    # Process and export each page
    print(f"\n[2] Exporting cleaned pages...\n")
    
    exported = 0
    skipped = 0
    total_raw = 0
    total_cleaned = 0
    
    # Create an index file
    index_content = "# Cleaned Pages Index\n\n"
    index_content += "| # | File | Title | Original | Cleaned | Reduction |\n"
    index_content += "|---|------|-------|----------|---------|----------|\n"
    
    for i, page in enumerate(pages, 1):
        url = page.get("url", "") or extract_page_url_from_content(page.get("markdown", ""))
        title = page.get("title", "Untitled")
        markdown = page.get("markdown", "")
        
        # Skip empty or error pages
        if not markdown or len(markdown) < 100 or "404" in title:
            skipped += 1
            continue
        
        # Apply cleaning
        cleaned = clean_markdown(markdown, noise_lines)
        
        # Skip if cleaning removed everything
        if len(cleaned) < 50:
            skipped += 1
            continue
        
        # Generate filename
        filename = sanitize_filename(title, url, i)
        filepath = OUTPUT_DIR / filename
        
        # Calculate stats
        raw_len = len(markdown)
        clean_len = len(cleaned)
        reduction = 100 - (clean_len / raw_len * 100) if raw_len > 0 else 0
        
        total_raw += raw_len
        total_cleaned += clean_len
        
        # Create page header
        page_header = f"# {title}\n\n"
        if url:
            page_header += f"**Source:** {url}\n\n"
        page_header += f"---\n\n"
        
        # Write the cleaned content
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(page_header + cleaned)
        
        exported += 1
        
        # Add to index
        index_content += f"| {i} | [{filename}](./{filename}) | {title[:30]}{'...' if len(title) > 30 else ''} | {raw_len:,} | {clean_len:,} | {reduction:.0f}% |\n"
        
        # Progress
        if i % 20 == 0:
            print(f"    Processed {i}/{len(pages)} pages...")
    
    # Write index file
    index_content += f"\n\n## Summary\n\n"
    index_content += f"- **Total pages processed:** {len(pages)}\n"
    index_content += f"- **Pages exported:** {exported}\n"
    index_content += f"- **Pages skipped:** {skipped}\n"
    index_content += f"- **Total raw content:** {total_raw:,} characters\n"
    index_content += f"- **Total cleaned content:** {total_cleaned:,} characters\n"
    index_content += f"- **Overall reduction:** {100 - (total_cleaned / total_raw * 100):.1f}%\n"
    
    with open(OUTPUT_DIR / "INDEX.md", "w", encoding="utf-8") as f:
        f.write(index_content)
    
    # Summary
    print(f"\n" + "=" * 70)
    print(f"EXPORT COMPLETE")
    print(f"=" * 70)
    print(f"Pages exported: {exported}")
    print(f"Pages skipped: {skipped}")
    print(f"Total raw content: {total_raw:,} characters")
    print(f"Total cleaned content: {total_cleaned:,} characters")
    print(f"Overall reduction: {100 - (total_cleaned / total_raw * 100):.1f}%")
    print(f"\nOutput: {OUTPUT_DIR.absolute()}")
    print(f"Index: {OUTPUT_DIR.absolute() / 'INDEX.md'}")


if __name__ == "__main__":
    main()
