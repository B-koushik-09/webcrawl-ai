"""
CollegeWeb AI - Text Cleaner Module
Removes navigation menus, headers, footers, and useless template text
from scraped webpage content before embedding.

Uses REPETITION-BASED detection instead of pattern-based:
- Lines that appear across many pages are automatically identified as boilerplate
- This catches navigation, headers, footers, menus that regex would miss
"""

import re
from typing import Optional, List, Dict, Set
from collections import Counter


# ========================================================
# GLOBAL NOISE LINES SET - Populated by build_noise_lines()
# ========================================================
_noise_lines: Set[str] = set()


def build_noise_lines(pages: List[Dict]) -> Set[str]:
    """
    Analyze all pages to find lines that appear repeatedly.
    Lines appearing in many pages are boilerplate (headers, footers, menus).
    
    Args:
        pages: List of page dicts with 'markdown' key
        
    Returns:
        Set of noise lines to remove
    """
    global _noise_lines
    
    counter = Counter()
    
    for page in pages:
        md = page.get("markdown", "") or page.get("content", "") or ""
        for line in md.split("\n"):
            line = line.strip()
            # Only count lines with substantial content
            if len(line) > 30:
                counter[line] += 1
    
    noise = set()
    # Lines appearing in more than 15 pages are likely boilerplate
    threshold = max(15, len(pages) // 10)  # At least 15, or 10% of pages
    
    for line, count in counter.items():
        if count > threshold:
            noise.add(line)
    
    _noise_lines = noise
    return noise


def get_noise_lines() -> Set[str]:
    """Get the current set of noise lines."""
    return _noise_lines


def clean_markdown(md: str, noise_lines: Optional[Set[str]] = None) -> str:
    """
    Clean markdown using repetition-based noise detection.
    
    Args:
        md: Raw markdown text
        noise_lines: Set of lines to remove (from build_noise_lines)
                    If None, uses global _noise_lines
    
    Returns:
        Cleaned markdown with meaningful content preserved
    """
    import re
    
    if not md:
        return ""
    
    if noise_lines is None:
        noise_lines = _noise_lines
    
    cleaned = []
    
    # Patterns to skip (navigation, menus, empty links)
    nav_patterns = [
        r'^-\s*\[.*?\]\(.*?#.*?\)$',  # Menu links with anchors
        r'^-\s*\[Notifications\]',     # Notification links
        r'^-\s*\[Home\]',              # Home links
        r'^-\s*\[Administration\]',    # Admin links
        r'^-\s*\[Academics\]',         # Academic links
        r'^-\s*\[Campus\]',            # Campus links
        r'^-\s*\[Campus Life\]',       # Campus Life links
        r'^\[\!\[.*?\]\(.*?\)\]\(.*?\)$',  # Image-only links
        r'^!\[.*?\]\(.*?\)$',          # Standalone images
        r'^-\s*$',                      # Empty list items
        r'^#+\s*$',                     # Empty headers
        r'^Close$',                     # Close buttons
        r'^Menu$',                      # Menu buttons
        r'^close$',
        r'^\|[\s\-\|]+\|$',            # Table separators with no content
    ]
    
    for line in md.split("\n"):
        line_stripped = line.strip()
        
        # Skip empty lines
        if not line_stripped:
            continue
        
        # Skip lines that appear in many pages (boilerplate)
        if line_stripped in noise_lines:
            continue
        
        # Skip navigation patterns
        skip_line = False
        for pattern in nav_patterns:
            if re.match(pattern, line_stripped, re.IGNORECASE):
                skip_line = True
                break
        if skip_line:
            continue
        
        # Skip very short lines (likely menu items, icons, etc.)
        # But keep headers (start with #), table rows (contain |), and list items with content
        is_header = line_stripped.startswith('#')
        is_table_row = '|' in line_stripped and len(line_stripped) > 10
        is_meaningful_list = line_stripped.startswith('-') and len(line_stripped) > 50
        
        if len(line_stripped) < 40 and not is_header and not is_table_row and not is_meaningful_list:
            continue
        
        # Skip lines that are ONLY links (no descriptive text)
        # Pattern: "[text](url)" where text is very short or just an image
        link_only_match = re.match(r'^\[([^\]]*)\]\([^\)]+\)$', line_stripped)
        if link_only_match:
            link_text = link_only_match.group(1)
            # Allow if the link text is substantial (explains what it is)
            if len(link_text) < 30 and not link_text.startswith('Click'):
                continue
        
        cleaned.append(line)
    
    result = "\n".join(cleaned)
    
    # Clean up excessive newlines
    result = re.sub(r'\n{3,}', '\n\n', result)
    
    return result.strip()


def clean_webpage_text(text: str, noise_lines: Optional[Set[str]] = None) -> str:
    """
    Removes navigation menus, headers, footers, and useless template text
    from scraped webpage content before embedding.
    
    Uses hybrid approach:
    1. Repetition-based: Remove lines that appear in many pages
    2. Pattern-based: Fall back to regex for edge cases
    
    Args:
        text: Raw webpage text (markdown or plain text)
        noise_lines: Optional set of noise lines from build_noise_lines()
    
    Returns:
        Cleaned text suitable for embedding
    """
    if not text:
        return ""
    
    # First, apply repetition-based cleaning if noise lines available
    if noise_lines or _noise_lines:
        text = clean_markdown(text, noise_lines)
    
    # Then apply pattern-based cleaning for remaining garbage
    text = re.sub(r'\s+', ' ', text)
    
    # ========================================================
    # FALLBACK GARBAGE PATTERNS - For edge cases regex can catch
    # ========================================================
    garbage_patterns = [
        # Footer patterns
        r'©.*?VNRVJIET',
        r'©.*?VNR VJIET',
        r'All rights reserved.*?$',
        r'Copyright.*?VNRVJIET',
        
        # Social media links (short mentions)
        r'\b(facebook|twitter|instagram|linkedin|youtube)\b',
        
        # Common header/navigation
        r'Skip to (main )?content',
        r'Toggle navigation',
        
        # Empty link text
        r'\[\s*\]\s*\([^)]+\)',
    ]
    
    for pattern in garbage_patterns:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    # ========================================================
    # LINE-LEVEL CLEANING
    # ========================================================
    lines = text.split('.')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Skip lines that are mostly links/menu junk
        link_count = len(re.findall(r'http|www|\.ac\.in|\.pdf|#\w+', line, re.IGNORECASE))
        if link_count > 3:
            continue
        
        # Skip very short lines (likely menu items)
        if len(line) < 30:
            continue
        
        cleaned_lines.append(line)
    
    result = '. '.join(cleaned_lines)
    
    # Final cleanup
    result = re.sub(r'\s+', ' ', result).strip()
    result = re.sub(r'\.+', '.', result)  # Multiple dots to single
    result = re.sub(r'\s+\.', '.', result)  # Space before dot
    
    return result


def clean_pdf_text(text: str) -> str:
    """
    Cleans extracted PDF text by removing common artifacts.
    
    Args:
        text: Raw PDF text
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove page numbers and headers/footers
    text = re.sub(r'Page\s*\d+\s*(of\s*\d+)?', '', text, flags=re.IGNORECASE)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common PDF artifacts
    text = re.sub(r'-\s*\n\s*', '', text)  # Hyphenated line breaks
    
    return text.strip()


if __name__ == "__main__":
    # Test the new repetition-based noise detection
    
    # Simulate multiple pages with repeated boilerplate
    test_pages = [
        {"markdown": """
Departments Automobile Engineering Chemistry Civil Engineering Computer Science
VNR VJIET is one of the premier engineering institutions in Telangana.
The college offers undergraduate and postgraduate programs in various disciplines.
Professional Chapters ACM ASME CSI IEEE ISTE SAE
Campus Celebrations Annual Day Convergence Cultural Day
© 2024 VNRVJIET. All rights reserved.
"""},
        {"markdown": """
Departments Automobile Engineering Chemistry Civil Engineering Computer Science
The Electronics department has state-of-the-art laboratories and equipment.
Faculty members have published over 500 research papers in international journals.
Professional Chapters ACM ASME CSI IEEE ISTE SAE
Campus Celebrations Annual Day Convergence Cultural Day
© 2024 VNRVJIET. All rights reserved.
"""},
        {"markdown": """
Departments Automobile Engineering Chemistry Civil Engineering Computer Science
Our placement cell has achieved 95% placement rate for the 2024 batch.
Top recruiters include Google, Microsoft, Amazon, and TCS.
Professional Chapters ACM ASME CSI IEEE ISTE SAE
Campus Celebrations Annual Day Convergence Cultural Day
© 2024 VNRVJIET. All rights reserved.
"""},
    ] * 10  # Multiply to simulate 30 pages
    
    print("=" * 60)
    print("REPETITION-BASED NOISE DETECTION TEST")
    print("=" * 60)
    
    # Step 1: Build noise lines from all pages
    noise_lines = build_noise_lines(test_pages)
    print(f"\n1. Found {len(noise_lines)} noise lines that appear across pages:")
    for line in sorted(noise_lines)[:5]:
        print(f"   - '{line[:60]}...'")
    
    # Step 2: Clean a single page using the noise lines
    test_page = test_pages[0]["markdown"]
    cleaned = clean_markdown(test_page, noise_lines)
    
    print(f"\n2. Original page ({len(test_page)} chars):")
    print(test_page[:200])
    
    print(f"\n3. Cleaned page ({len(cleaned)} chars):")
    print(cleaned if cleaned else "   (All content was identified as boilerplate)")
    
    print("\n" + "=" * 60)
    print("The key insight: Lines like 'Departments Automobile Engineering...'")
    print("are NOW removed because they appear in 30 pages, not because of regex!")
    print("=" * 60)
