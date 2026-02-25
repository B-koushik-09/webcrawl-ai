"""
Faculty PDF Extractor - Extract structured faculty data from PDFs

This script:
1. Reads all faculty PDFs from data/pdfs/
2. Extracts: Name, Designation, Department, Experience, Education
3. Appends structured data to data/faculty.md
4. Optionally deletes processed faculty PDFs

Usage:
    python extract_faculty.py
    python extract_faculty.py --delete-pdfs  # Also delete faculty PDFs after extraction
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional

try:
    import pypdf
except ImportError:
    print("Installing pypdf...")
    os.system("pip install pypdf")
    import pypdf

# Suppress pypdf warnings about malformed PDFs
import logging
logging.getLogger("pypdf").setLevel(logging.ERROR)


# Configuration
PDF_DIR = Path(__file__).parent / "data" / "pdfs"
OUTPUT_FILE = Path(__file__).parent / "data" / "faculty.md"

# Keywords that identify faculty PDFs
FACULTY_PDF_KEYWORDS = [
    "dr-", "dr.", "dr ", "prof-", "prof.", "prof ", 
    "mr.", "mrs.", "ms.", "faculty", "staff", "hod",
    "principal", "dean", "director",
    "profile", "cv", "resume", "biodata",
    "teaching", "academic"
]

# Patterns to extract faculty information
PATTERNS = {
    "name": [
        r"(?:Name|Dr\.|Prof\.|Mr\.|Mrs\.|Ms\.)\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
        r"^([A-Z][A-Z\s]+)$",  # All caps name at start
    ],
    "designation": [
        r"(?:Designation|Position|Title)\s*[:\-]?\s*([^\n]+)",
        r"(Professor|Associate Professor|Assistant Professor|HOD|Head of Department|Dean|Principal|Lecturer)[^\n]*",
    ],
    "department": [
        r"(?:Department|Dept|Branch)\s*[:\-]?\s*([^\n]+)",
        r"(Computer Science|Electronics|Mechanical|Civil|Electrical|IT|Information Technology|ECE|EEE|CSE|MBA|MCA)[^\n]*",
    ],
    "experience": [
        r"(?:Experience|Years of Experience|Teaching Experience)\s*[:\-]?\s*([^\n]+)",
        r"(\d+)\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|teaching)",
    ],
    "education": [
        r"(?:Education|Qualification|Degree|Degrees)\s*[:\-]?\s*([^\n]+)",
        r"(Ph\.?D\.?|M\.?Tech\.?|M\.?E\.?|B\.?Tech\.?|B\.?E\.?|MBA|MCA|M\.?Sc\.?|B\.?Sc\.?)[^\n,]*",
    ],
    "email": [
        r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
    ],
    "specialization": [
        r"(?:Specialization|Area of Interest|Research Area|Research Interest)\s*[:\-]?\s*([^\n]+)",
    ],
}

BLACKLIST_KEYWORDS = [
    "syllabus", "curriculum", "calendar", "brochure", "placement", 
    "report", "regulation", "handbook", "timetable", "exam", 
    "fee", "payment", "scholarship", "internship", "project", 
    "mou", "patent", "policy", "guideline", "form", "poster",
    "highlight", "statist", "naac", "nba", "audit"
]

def is_faculty_pdf(filename: str) -> bool:
    """
    Check if PDF is likely a faculty profile using heuristics.
    1. Contains faculty title (Dr, Prof)
    2. Looks like a name (e.g. '12.Name.pdf', 'Name Surname.pdf')
    3. Excludes common non-faculty words (syllabus, brochure)
    """
    name_lower = filename.lower()
    
    # 1. Immediate rejection
    if any(bad in name_lower for bad in BLACKLIST_KEYWORDS):
        return False
        
    # 2. Strong keywords (Dr, Prof, CV, Profile)
    if any(keyword in name_lower for keyword in FACULTY_PDF_KEYWORDS):
        return True
        
    # 3. Pattern match for "Number.Name.pdf" or "Name.pdf"
    # Examples: "12.Anjusha_CSE.pdf", "14.Mrs.E.Lalitha.pdf", "Haritha-N.pdf"
    
    # Clean filename: remove extension, numbers at start
    clean_name = re.sub(r'\.pdf$', '', filename, flags=re.IGNORECASE)
    clean_name = re.sub(r'^\d+[\s.\-_]+', '', clean_name) # Remove "12."
    
    # If remaining part looks like a name (2-4 words, mostly letters)
    # Allow dots, hyphens, spaces
    words = re.split(r'[\s.\-_]+', clean_name)
    valid_words = [w for w in words if len(w) > 1 and w.isalpha()]
    
    # A typical name has 2-4 parts (First Last, First Middle Last)
    if 1 <= len(valid_words) <= 4:
        # Check if it doesn't contain weird characters
        if re.match(r'^[a-zA-Z\s.\-_]+$', clean_name):
            return True
            
    return False

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file."""
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        print(f"  [ERROR] Failed to read {pdf_path.name}: {e}")
        return ""


def extract_field(text: str, field_name: str) -> Optional[str]:
    """Extract a field using multiple regex patterns."""
    patterns = PATTERNS.get(field_name, [])
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1).strip()
            # Clean up the value
            value = re.sub(r'\s+', ' ', value)  # Normalize whitespace
            value = value[:200]  # Limit length
            if len(value) > 3:  # Minimum valid length
                return value
    
    return None


def extract_faculty_info(text: str, filename: str) -> Dict[str, str]:
    """Extract all faculty information from PDF text."""
    info = {
        "name": extract_field(text, "name"),
        "designation": extract_field(text, "designation"),
        "department": extract_field(text, "department"),
        "experience": extract_field(text, "experience"),
        "education": extract_field(text, "education"),
        "email": extract_field(text, "email"),
        "specialization": extract_field(text, "specialization"),
        "source_file": filename,
    }
    
    # Try to extract name from filename if not found in text
    if not info["name"]:
        # Pattern: "1-Dr-Name-Here.pdf" or "Dr-Name-Here.pdf"
        name_match = re.search(r'(?:\d+-)?(?:Dr|Prof|Mr|Mrs|Ms)?-?([A-Za-z\-]+(?:-[A-Za-z]+)*)\.pdf', filename, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).replace('-', ' ').title()
            info["name"] = name
    
    return info


def format_faculty_entry(info: Dict[str, str]) -> str:
    """Format faculty info as markdown entry."""
    lines = []
    
    name = info.get("name") or "Unknown Faculty"
    lines.append(f"### {name}")
    lines.append("")
    
    if info.get("designation"):
        lines.append(f"**Designation:** {info['designation']}")
    
    if info.get("department"):
        lines.append(f"**Department:** {info['department']}")
    
    if info.get("experience"):
        lines.append(f"**Experience:** {info['experience']}")
    
    if info.get("education"):
        lines.append(f"**Education:** {info['education']}")
    
    if info.get("email"):
        lines.append(f"**Email:** {info['email']}")
    
    if info.get("specialization"):
        lines.append(f"**Specialization:** {info['specialization']}")
    
    lines.append(f"*Source: {info.get('source_file', 'Unknown')}*")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    return "\n".join(lines)


def main(delete_pdfs: bool = False):
    """Main extraction function."""
    print("=" * 60)
    print("Faculty PDF Extractor")
    print("=" * 60)
    
    if not PDF_DIR.exists():
        print(f"[ERROR] PDF directory not found: {PDF_DIR}")
        return
    
    # Find faculty PDFs
    all_pdfs = list(PDF_DIR.glob("*.pdf"))
    faculty_pdfs = [p for p in all_pdfs if is_faculty_pdf(p.name)]
    
    print(f"\n[INFO] Found {len(all_pdfs)} total PDFs")
    print(f"[INFO] Found {len(faculty_pdfs)} faculty PDFs")
    
    if not faculty_pdfs:
        print("[WARNING] No faculty PDFs found!")
        return
    
    # Create/clear output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("# VNR VJIET Faculty Directory\n\n")
        f.write("This file contains structured faculty information extracted from PDF profiles.\n\n")
        f.write("---\n\n")
    
    # Process each faculty PDF
    extracted_count = 0
    deleted_pdfs = []
    
    for pdf_path in faculty_pdfs:
        print(f"\n[PROCESSING] {pdf_path.name}")
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f"  [SKIP] No text extracted")
            continue
        
        # Extract faculty info
        info = extract_faculty_info(text, pdf_path.name)
        
        # Check if we got useful info
        if info.get("name") or info.get("designation"):
            # Format and append to file
            entry = format_faculty_entry(info)
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(entry)
            
            print(f"  [OK] Extracted: {info.get('name', 'Unknown')}")
            extracted_count += 1
            
            # Mark for deletion if requested
            if delete_pdfs:
                deleted_pdfs.append(pdf_path)
        else:
            print(f"  [SKIP] No useful data found")
    
    # Delete processed PDFs if requested
    if delete_pdfs and deleted_pdfs:
        print(f"\n[INFO] Deleting {len(deleted_pdfs)} processed faculty PDFs...")
        for pdf_path in deleted_pdfs:
            try:
                pdf_path.unlink()
                print(f"  [DELETED] {pdf_path.name}")
            except Exception as e:
                print(f"  [ERROR] Failed to delete {pdf_path.name}: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Faculty profiles extracted: {extracted_count}")
    print(f"Output file: {OUTPUT_FILE}")
    if delete_pdfs:
        print(f"PDFs deleted: {len(deleted_pdfs)}")
    print("\nNext steps:")
    print("1. Review data/faculty.md")
    print("2. Run indexing to include faculty data")


if __name__ == "__main__":
    import sys
    delete = "--delete-pdfs" in sys.argv
    main(delete_pdfs=delete)
