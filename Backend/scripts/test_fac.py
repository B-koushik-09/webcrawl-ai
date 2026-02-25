import os
import re

path = r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\faculty.md"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

patterns = ["Designation", "Experience", "Education", "Teaching Interests", "Academic Contribution", "Papers published", "Papers presented"]
for p in patterns:
    content = re.sub(rf'(\*\*{p}:?\*\*)\s*', r'\n## \1\n\1 ', content)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("TEST_FACULTY_SUCCESS")
