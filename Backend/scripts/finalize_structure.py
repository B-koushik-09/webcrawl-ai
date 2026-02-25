import os
import re

SYLLABUS_FILES = [
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_I_Year.md",
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_II_Year.md",
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_III_Year.md",
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_IV_Year.md"
]

HOSTEL_FILES = [
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\074_hostel.md",
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_VNRVJIET_Hostel_Brochure_2025_26.md"
]

def improve_syllabus(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    matches = re.findall(r'(\bUNIT-[IVX1-5]+\b:?\s*)', content)
    print(f"File: {os.path.basename(path)}, Found {len(matches)} units.")
    
    # 1. Convert UNIT-I etc. to headings
    content = re.sub(r'(\bUNIT-[IVX1-5]+\b:?\s*)', r'\n## \1', content)
    
    # 2. Cleanup
    content = content.replace("## ##", "##")
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Improved syllabus structure: {path}")

def improve_hostel(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if "Brochure" in path:
        content = re.sub(r'(FEE\s+DETAILS)', r'\n## \1', content)
        content = re.sub(r'(CONTACT\s+DETAILS)', r'\n## \1', content)
        content = content.replace("## ##", "##")
    else:
        if "## Fees & Brochure" not in content:
            content = content.replace("Hostel Brochure", "## Fees & Brochure\nHostel Brochure")
        if "## Rules" not in content:
            content = content.replace("Rules And Regulations", "## Rules\nRules And Regulations")
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Improved hostel structure: {path}")

if __name__ == "__main__":
    for f in SYLLABUS_FILES:
        improve_syllabus(f)
    for f in HOSTEL_FILES:
        improve_hostel(f)
