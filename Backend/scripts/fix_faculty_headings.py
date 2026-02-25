import re
import os

file_path = r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\faculty.md"
if not os.path.exists(file_path):
    print("File not found")
    exit(1)

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

patterns = ["Designation", "Experience", "Education", "Teaching Interests", "Academic Contribution", "Papers published", "Papers presented"]
for p in patterns:
    # Match bolded or unbolded keys at start of line
    # Using \r? to handle Windows line endings if needed
    regex = rf'^(?:\*\*)?({p}.*?:)(?:\*\*)?.*$'
    content = re.sub(regex, r'## \1\n\g<0>', content, flags=re.MULTILINE)

# Deduplicate headings if run twice
content = re.sub(r'(## .*?\n)\1', r'\1', content)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Successfully updated faculty.md")
