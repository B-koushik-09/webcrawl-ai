import os
import re

path = r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_I_Year.md"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

delim = "VNR VIGNANA JYOTHI INSTITUTE OF ENGINEERING AND TECHNOLOGY"
count = content.count(delim)
print(f"Found {count} occurrences of delimiter.")

# Split by delimiter
parts = content.split(delim)
new_parts = [parts[0]] # Header info

for p in parts[1:]:
    # In each part, the first thing is usually (CODE) TITLE
    # Try to find (22XXXXXXX) 
    code_match = re.search(r'\((\b22[A-Z0-9]{8})\)', p)
    if code_match:
        code = code_match.group(1)
        # Title is usually between the code and the next keyword
        # We'll just grab the next 50 chars and clean it up
        title_area = p[code_match.end():code_match.end()+100]
        title = re.split(r'TEACHING|COURSE|UNIT|EVALUATION', title_area)[0].strip()
        print(f"  Fixing: {title} ({code})")
        
        # Prepend heading
        p = f"\n\n---\n\n## {title} ({code})\n\n" + delim + p
    else:
        p = delim + p
    
    # Also fix UNIT-X on the way
    p = re.sub(r'\b(UNIT-[I|V|X]+)\b', r'\n\n## \1\n\n', p)
    new_parts.append(p)

new_content = "".join(new_parts)
with open(path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print("DONE.")
