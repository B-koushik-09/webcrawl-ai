"""
Convert the 4 CSE R22 syllabus PDFs from data/ to cleaned markdown in cleaned_pages/.
Follows the same format as export_pdfs_to_cleaned.py.
"""
import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())

from config import BASE_DIR, DATA_DIR
from pypdf import PdfReader
from modules.text_cleaner import clean_webpage_text
import logging

logging.getLogger("pypdf").setLevel(logging.ERROR)

# The 4 new CSE R22 PDFs in data/ root
CSE_PDFS = [
    "I B.Tech. CSE.pdf",
    "CSE-II-YEAR-R22.pdf",
    "5-B.Tech-CSE.pdf",
    "5. CSE - R22 @ 19.07.2025.pdf",
]

# Descriptive output names for clarity
OUTPUT_NAMES = {
    "I B.Tech. CSE.pdf": "PDF_CSE_R22_I_Year",
    "CSE-II-YEAR-R22.pdf": "PDF_CSE_R22_II_Year",
    "5-B.Tech-CSE.pdf": "PDF_CSE_R22_III_Year",
    "5. CSE - R22 @ 19.07.2025.pdf": "PDF_CSE_R22_IV_Year",
}

def convert():
    cleaned_dir = BASE_DIR / "cleaned_pages"
    cleaned_dir.mkdir(exist_ok=True)
    
    count = 0
    for pdf_name in CSE_PDFS:
        pdf_path = DATA_DIR / pdf_name
        if not pdf_path.exists():
            print(f"[SKIP] Not found: {pdf_name}")
            continue
        
        try:
            reader = PdfReader(pdf_path)
            text_pages = []
            
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_pages.append(text)
            
            if not text_pages:
                print(f"[SKIP] No text: {pdf_name}")
                continue
            
            full_text = "\n\n".join(text_pages)
            
            # Clean the text
            full_text = clean_webpage_text(full_text)
            
            # Build markdown
            out_name = OUTPUT_NAMES.get(pdf_name, f"PDF_{pdf_name.replace(' ', '_')}")
            md_content = f"""# {out_name.replace('_', ' ').replace('PDF ', '')}

**Source:** PDF Document ({pdf_name})
**Department:** CSE
**Regulation:** R22

---

{full_text}
"""
            out_path = cleaned_dir / f"{out_name}.md"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            
            print(f"[OK] {pdf_name} -> {out_name}.md ({len(text_pages)} pages, {len(full_text)} chars)")
            count += 1
            
        except Exception as e:
            print(f"[ERROR] {pdf_name}: {e}")
    
    print(f"\n[DONE] Converted {count} CSE R22 syllabus PDFs to cleaned_pages/")
    print(f"\nNext steps:")
    print(f"  1. Run: python trigger_rebuild.py    (to re-index with new files)")
    print(f"  2. Run: python migrate_dept.py        (to re-tag dept metadata)")
    print(f"  3. Run: python test_cse_retrieval.py   (to verify)")

if __name__ == "__main__":
    convert()
