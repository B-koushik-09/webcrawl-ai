
import os

files = [
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_I_Year.md",
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_II_Year.md",
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_III_Year.md",
    r"c:\VNRVJIET\Projects\ai-chat\Backend\cleaned_pages\PDF_CSE_R22_IV_Year.md"
]

for f in files:
    if os.path.exists(f):
        with open(f, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            print(f"{os.path.basename(f)}: {len(lines)} lines", flush=True)
    else:
        print(f"File not found: {f}", flush=True)
print("Script finished.", flush=True)
