
import os
import re
import sys

# Define base directory
BASE_DIR = r"c:\VNRVJIET\Projects\ai-chat\Backend"
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned_pages")
LOG_FILE = os.path.join(BASE_DIR, "refine_log.txt")

with open(LOG_FILE, "w") as log:
    log.write("Script started.\n")

files_to_process = [
    "PDF_CSE_R22_I_Year.md",
    "PDF_CSE_R22_II_Year.md",
    "PDF_CSE_R22_III_Year.md",
    "PDF_CSE_R22_IV_Year.md"
]

# Regex to find (CODE) SUBJECT
# Matches: (22BS1MT101) MATRICES AND CALCULUS
# Also: B. I Semester (22BS1MT101) MATRICES...
header_pattern = re.compile(r"\(([A-Z0-9]{8,})\)\s+([A-Z\s\-\&]+)")

def log_msg(msg):
    print(msg)
    with open(LOG_FILE, "a") as log:
        log.write(msg + "\n")

def process_file(filename):
    filepath = os.path.join(CLEANED_DIR, filename)
    if not os.path.exists(filepath):
        log_msg(f"File not found: {filepath}")
        return

    log_msg(f"Processing {filename}...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        log_msg(f"Error reading {filename}: {e}")
        return

    new_lines = []
    modified_count = 0
    
    for line in lines:
        line_stripped = line.strip()
        
        if line_stripped.startswith("##"):
            new_lines.append(line)
            continue
            
        match = header_pattern.search(line_stripped)
        
        if match:
            code = match.group(1)
            subject = match.group(2).strip()
            
            if len(subject) > 3 and subject.isupper():
                if "TEACHING SCHEME" in line_stripped or "EVALUATION SCHEME" in line_stripped:
                     new_lines.append(line)
                     continue

                log_msg(f"  Adding Header: {line_stripped}")
                new_lines.append(f"## {line_stripped}\n")
                modified_count += 1
            else:
                new_lines.append(line)
        else:
             new_lines.append(line)

    if modified_count > 0:
        log_msg(f"  Modified {modified_count} lines in {filename}.")
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
        log_msg(f"  Saved {filename}.")
    else:
        log_msg(f"  No changes needed for {filename}.")

if __name__ == "__main__":
    for f in files_to_process:
        process_file(f)
    log_msg("Done.")
