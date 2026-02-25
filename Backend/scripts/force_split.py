import os

path = r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_I_Year.md"
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace "VNR VIGNANA JYOTHI" with "\r\n---\r\n\r\n# VNR VIGNANA JYOTHI"
# This is a crude but effective way to force line breaks
new_content = content.replace("VNR VIGNANA JYOTHI", "\r\n\r\n---\r\n\r\n# VNR VIGNANA JYOTHI")

# Also split UNIT-I
new_content = new_content.replace("UNIT-I", "\r\n\r\n## UNIT-I\r\n\r\n")
new_content = new_content.replace("UNIT-II", "\r\n\r\n## UNIT-II\r\n\r\n")
new_content = new_content.replace("UNIT-III", "\r\n\r\n## UNIT-III\r\n\r\n")
new_content = new_content.replace("UNIT-IV", "\r\n\r\n## UNIT-IV\r\n\r\n")
new_content = new_content.replace("UNIT-V", "\r\n\r\n## UNIT-V\r\n\r\n")

with open(path, 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(new_content)

print(f"FORCED SPLIT DONE. Length changed from {len(content)} to {len(new_content)}")
