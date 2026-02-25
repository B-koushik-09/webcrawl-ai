import os
import re
from pathlib import Path

# Targeted files list (as requested)
TARGET_FILES = [
    # CSE Syllabus
    "PDF_CSE_R22_I_Year.md",
    "PDF_CSE_R22_II_Year.md",
    "PDF_CSE_R22_III_Year.md",
    "PDF_CSE_R22_IV_Year.md",
    
    # CSE Webpages
    "060_cse.md",
    "129_cse-aiml-and-iot.md",
    "038_cse-ds-and-cys.md",
    
    # Faculty
    "faculty.md",
    
    # Hostel
    "074_hostel.md",
    "PDF_VNRVJIET_Hostel_Brochure_2025_26.md",
    "PDF_Application_for_Admission_in_the_Hostel_of__VNRVJIET.md"
]

CLEANED_PAGES_DIR = Path(r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages")

def improve_syllabus(content):
    """Adds ## headings for Units and Course Titles in syllabus files."""
    # Pattern for UNIT-I, UNIT-II, etc.
    content = re.sub(r'^(UNIT-[IVX1-5]+:?.*?)$', r'## \1', content, flags=re.MULTILINE)
    
    # Pattern for course titles in specific Semester blocks
    # e.g. "3 VNR ... BIG DATA ANALYTICS"
    content = re.sub(r'^\d+\s+VNR\s+.*?\(.*?\)\s+([A-Z\s]{5,})$', r'## \1', content, flags=re.MULTILINE)
    
    return content

def improve_faculty(content):
    """Adds ## headings for faculty sections."""
    # Pattern for numbered sections like "1. Educational..." (can be bolded)
    content = re.sub(r'^(?:\*\*)?(\d+\.\s+[A-Z][a-z\s]+:?)(?:\*\*)?$', r'## \1', content, flags=re.MULTILINE)

    # Pattern for "Designation:", "Experience:", etc. 
    # Handle optional bolding and trailing content
    patterns = ["Designation", "Experience", "Educational / Technical qualifications",
                "Teaching Interests", "Co-curricular", "Academic Contribution", 
                "Papers published", "Papers presented", "Projects"]

    for p in patterns:
        # If line starts with **Key:** or Key:
        content = re.sub(f'^(?:\*\*)?({p}.*?:)(?:\*\*)?.*$', r'## \1\n\g<0>', content, flags=re.MULTILINE)

    return content

def improve_hostel(content):
    """Adds ## headings for hostel sections."""
    patterns = ["FEE STRUCTURE", "RULES AND REGULATIONS", "ADMISSION PROCESS", "FACILITIES", "GENERAL HINTS"]
    for p in patterns:
        # Case insensitive match, handle optional bolding
        content = re.sub(f'(?i)^(?:\\*\\*)?({p}.*?)(?:\\*\\*)?$', r'## \1', content, flags=re.MULTILINE)
    return content

def process_file(file_name):
    file_path = CLEANED_PAGES_DIR / file_name
    if not file_path.exists():
        print(f"[SKIP] File not found: {file_name}")
        return
    
    print(f"[PROCESS] Improving headings in {file_name}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Apply heading logic based on file type
    if "CSE_R22" in file_name:
        content = improve_syllabus(content)
    elif "faculty" in file_name or file_name.startswith("PDF_"):
        content = improve_faculty(content)
    elif "hostel" in file_name.lower():
        content = improve_hostel(content)
    
    # Ensure there's a ## heading at the top (after the main # title)
    # This helps chunking if the file has no other headings
    lines = content.split('\n')
    has_double_heading = any(line.startswith('## ') for line in lines)
    
    if not has_double_heading and len(lines) > 5:
        # Find the first significant text after metadata and add a heading
        for i, line in enumerate(lines):
            if i > 5 and line.strip() and not line.startswith('#') and not line.startswith('---') and not line.startswith('*'):
                lines[i] = f"## Introduction\n{line}"
                break
        content = '\n'.join(lines)

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[OK] Updated {file_name}")
    else:
        print(f"[INFO] no changes needed for {file_name}")

if __name__ == "__main__":
    for file_name in TARGET_FILES:
        process_file(file_name)
    
    # Also look for any other CSE syllabus files or faculty profiles that might be missed
    for file_path in CLEANED_PAGES_DIR.glob("PDF_CSE_*.md"):
        if file_path.name not in TARGET_FILES:
            process_file(file_path.name)
            
    print("[DONE] Targeted heading improvement complete.")
