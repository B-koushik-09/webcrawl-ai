import re
import os

file_path = r'c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_III_Year.md'

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Standardize Section Headers
# Replace bold headers with Level 4 Markdown headers
content = re.sub(r'\*\*COURSE OBJECTIVES:?\s*\*\*', '#### COURSE OBJECTIVES', content, flags=re.IGNORECASE)
content = re.sub(r'\*\*COURSE OUTCOMES:?\s*\*\*', '#### COURSE OUTCOMES', content, flags=re.IGNORECASE)
content = re.sub(r'\*\*SYLLABUS\*\*', '#### SYLLABUS', content, flags=re.IGNORECASE)
content = re.sub(r'\*\*TEXT\s*BOOKS:?\s*\*\*', '#### TEXT BOOKS', content, flags=re.IGNORECASE)
content = re.sub(r'\*\*REFERENCES:?\s*\*\*', '#### REFERENCES', content, flags=re.IGNORECASE)
content = re.sub(r'\*\*WEB\s*RESOURCES:?\s*\*\*', '#### WEB RESOURCES', content, flags=re.IGNORECASE)

# Standardize Unit Headers
# Pattern 1: **UNIT-III: Partitioning** -> #### UNIT-III\n**Partitioning**:
# Regex to capture UNIT number and Title
def replace_unit_header(match):
    unit_num = match.group(1)
    title = match.group(2).strip()
    return f'#### UNIT-{unit_num}\n**{title}**:'

# Handling **UNIT-X: Title**
content = re.sub(r'\*\*UNIT-([IVX]+)[:\s]+(.*?)\*\*', replace_unit_header, content)

# Handling #### UNIT-X: Title
content = re.sub(r'#{4}\s*UNIT-([IVX]+)[:\s]+(.*?)$', replace_unit_header, content, flags=re.MULTILINE)

# Handling #### UNIT-X (newline) **Title:** (Ensure title is bold with colon)
# This might already be correct, but let's normalize. 
# If we have #### UNIT-I followed by **Title:**, we leave it, but ensure it's standardized.
# Actually, the replace_unit_header logic above handles the single-line cases. 
# The multi-line case `#### UNIT-I\n**Title:**` is acceptable as per plan? 
# Plan says:
# #### UNIT-[N]
# **[Unit Title]**: [Unit Content]
# So `#### UNIT-I` followed by `**Title**:` is good.

# Search for any remaining `UNIT-` headers that might be malformatted.
# e.g. `**UNIT-I**` without title on same line?
# Line 2809: `#### UNIT-I` then `**Robot Programming:**`
# This is fine.

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Headers standardized successfully.")
