import os
import re

TARGET_FILES = [
    "PDF_CSE_R22_I_Year.md", "PDF_CSE_R22_II_Year.md", "PDF_CSE_R22_III_Year.md", "PDF_CSE_R22_IV_Year.md",
    "060_cse.md", "129_cse-aiml-and-iot.md", "038_cse-ds-and-cys.md",
    "faculty.md",
    "074_hostel.md", "PDF_VNRVJIET_Hostel_Brochure_2025_26.md", "PDF_Application_for_Admission_in_the_Hostel_of__VNRVJIET.md"
]

BASE_DIR = r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages"

def process_syllabus(content):
    # Match UNIT-I etc
    content = re.sub(r'(UNIT-[IVX1-5]+:?\s+)', r'\n## \1', content)
    # Match VNR VIGNANA... headings
    content = re.sub(r'(VNR VIGNANA JYOTHI INSTITUTE OF ENGINEERING AND TECHNOLOGY)', r'\n## \1', content)
    return content

def process_faculty(content):
    # Match bolded keys even if they have spaces or are buried
    patterns = ["Designation", "Experience", "Education", "Teaching Interests", "Academic Contribution", "Papers published", "Papers presented"]
    for p in patterns:
        content = re.sub(rf'(\*\*{p}:?\*\*)\s*', r'\n## \1\n\1 ', content)
    return content

def process_hostel(content):
    patterns = ["FEE STRUCTURE", "RULES AND REGULATIONS", "ADMISSION PROCESS", "FACILITIES", "GENERAL HINTS"]
    for p in patterns:
        content = re.sub(rf'(?i)({p})', r'\n## \1\n', content)
    return content

for file_name in TARGET_FILES:
    path = os.path.join(BASE_DIR, file_name)
    if not os.path.exists(path):
        print(f"Skipping {file_name} (not found)")
        continue
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    if "CSE_R22" in file_name:
        content = process_syllabus(content)
    elif "faculty" in file_name or file_name.startswith("PDF_"):
        content = process_faculty(content)
    elif "hostel" in file_name.lower():
        content = process_hostel(content)
        
    if content != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {file_name}")
    else:
        print(f"No changes for {file_name}")

print("COMPLETED_PROCESS")
