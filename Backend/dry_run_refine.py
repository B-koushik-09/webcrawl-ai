
import os
import re
import sys

cleaned_dir = r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages"
files_to_process = [
    "PDF_CSE_R22_I_Year.md",
    "PDF_CSE_R22_II_Year.md",
    "PDF_CSE_R22_III_Year.md",
    "PDF_CSE_R22_IV_Year.md"
]

# Regex to find (CODE) SUBJECT
# Matches: (22BS1MT101) MATRICES AND CALCULUS
# Also: B. I Semester (22BS1MT101) MATRICES...
header_pattern = re.compile(r"\(([A-Z0-9]{8,})\)\s+([A-Z\s\-\&]+)")

def process_file(filename):
    filepath = os.path.join(cleaned_dir, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filename}")
        return

    print(f"Processing {filename}...", flush=True)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return

    modified_count = 0
    
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        match = header_pattern.search(line_stripped)
        
        if match:
            # Check if it already has a header
            if line_stripped.startswith("#"):
                continue
                
            code = match.group(1)
            subject = match.group(2).strip()
            
            # Filter out lines that are just TOC or not subjects
            # Subjects usually are uppercase.
            if len(subject) > 3 and subject.isupper():
                print(f"  [MATCH] {line_stripped}")
                print(f"       -> ## {line_stripped}")
                modified_count += 1
                
    if modified_count == 0:
        print(f"  No matches found in {filename}.")
    else:
        print(f"  Found {modified_count} potential headers.")

if __name__ == "__main__":
    for f in files_to_process:
        process_file(f)
