import os
import re
import sys

FILES = [
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_I_Year.md",
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_II_Year.md",
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_III_Year.md",
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_IV_Year.md"
]

def log(msg):
    print(msg)
    sys.stdout.flush()

def sync_year(path):
    if not os.path.exists(path):
        log(f"SKIP: {path} not found")
        return
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Force split at common delimiters
    # College Name often preceded by a page number
    delim_regex = r'(\d+)\s+VNR\s+VIGNANA\s+JYOTHI\s+INSTITUTE\s+OF\s+ENGINEERING\s+AND\s+TECHNOLOGY'
    new_content = re.sub(delim_regex, r'\n\n---\n\n### [Page \1] VNR VIGNANA JYOTHI INSTITUTE OF ENGINEERING AND TECHNOLOGY\n\n', content)
    
    # Unit markers: UNIT-I, UNIT-II, etc.
    unit_regex = r'\b(UNIT-[IVXLC]+)\b'
    new_content = re.sub(unit_regex, r'\n\n## \1\n\n', new_content)
    
    # 2. Insert Subject Headings
    # In Semester tables: 22BS1MT101 Matrices and Calculus
    # In body: (22BS1MT101) MATRICES AND CALCULUS
    def subject_replacer(match):
        code = match.group(1)
        title = match.group(2).strip()
        # Clean title
        title = re.split(r'TEACHING|COURSE|UNIT|EVALUATION|SCHEME|PRE-REQUISITES', title)[0].strip()
        return f"\n\n## {title} ({code})\n\n" + match.group(0)

    # Subject start pattern: (CODE) TITLE
    subject_pattern = r'\((\b22[A-Z0-9]{8})\)\s+([A-Z\s&/,\-]{5,100})'
    new_content = re.sub(subject_pattern, subject_replacer, new_content)
    
    # cleanup:
    new_content = re.sub(r'\n{3,}', '\n\n', new_content)
    
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(new_content)
    
    # Verify
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        log(f"SYNCED: {os.path.basename(path)} | Lines: {len(lines)}")

if __name__ == "__main__":
    for f in FILES:
        sync_year(f)
