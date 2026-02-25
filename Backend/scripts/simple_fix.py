import os
import re

SYLLABUS_FILES = [
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_I_Year.md",
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_II_Year.md",
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_III_Year.md",
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_IV_Year.md"
]

def process(path):
    if not os.path.exists(path):
        print(f"NOT FOUND: {path}")
        return
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Simple replacement for UNIT headings
    original_len = len(content)
    # Match UNIT-I, UNIT-II, etc.
    content = re.sub(r'(UNIT-[IVX1-5]+)', r'\n## \1', content)
    
    # Cleanup double headings
    content = content.replace("## ##", "##")
    
    if len(content) != original_len:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"UPDATED: {os.path.basename(path)} (Size changed: {original_len} -> {len(content)})")
    else:
        print(f"NO CHANGE: {os.path.basename(path)}")

if __name__ == "__main__":
    for f in SYLLABUS_FILES:
        process(f)
