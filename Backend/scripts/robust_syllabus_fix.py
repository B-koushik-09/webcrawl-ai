import os
import re

FILES = [
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_I_Year.md",
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_II_Year.md",
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_III_Year.md",
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_IV_Year.md"
]

def fix_syllabus(path):
    if not os.path.exists(path):
        print(f"ERROR: {path} not found")
        return
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Improved regex to find UNIT-I, UNIT-II etc and prepend ## 
    # This also handles cases where it might already have one # or be in the middle of a line
    # We use \b to match word boundary, and handle colon/space
    
    patterns = [
        r'UNIT-I', r'UNIT-II', r'UNIT-III', r'UNIT-IV', r'UNIT-V',
        r'UNIT I', r'UNIT II', r'UNIT III', r'UNIT IV', r'UNIT V'
    ]
    
    changed = False
    for p in patterns:
        find_pattern = r'(?<!## )' + p  # Not preceded by ##
        new_content = re.sub(find_pattern, r'\n\n## ' + p, content)
        if new_content != content:
            content = new_content
            changed = True
    
    if changed:
        # Cleanup: sometimes PDFs have "UNIT-I: UNIT-I" or similar
        content = content.replace("## ##", "##")
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"SUCCESS: Applied headings to {os.path.basename(path)}")
    else:
        print(f"INFO: No matching UNIT markers needed fixing in {os.path.basename(path)}")

if __name__ == "__main__":
    for f in FILES:
        fix_syllabus(f)
