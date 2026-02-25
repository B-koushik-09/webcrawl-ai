import sys
import os
from pathlib import Path

# Add current dir to path
sys.path.append(os.getcwd())

from config import PDF_DIR, BASE_DIR
from pypdf import PdfReader
from modules.text_cleaner import clean_webpage_text
import logging

# Suppress warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)

def export_pdfs_to_md():
    cleaned_dir = BASE_DIR / "cleaned_pages"
    cleaned_dir.mkdir(exist_ok=True)
    
    if not PDF_DIR.exists():
        print(f"[ERROR] PDF directory not found: {PDF_DIR}")
        return

    pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"[*] Found {len(pdf_files)} PDFs to export.")
    
    count = 0
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            text_content = []
            
            # Extract text
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_content.append(text)
            
            if not text_content:
                print(f"[SKIP] No text found in {pdf.name}")
                continue
                
            full_text = "\n\n".join(text_content)
            
            # Apply Cleaning
            full_text = clean_webpage_text(full_text)
            
            # Create Markdown format
            # Prefix filename with 999_ to make it load last? Or generic PDF_
            safe_name = pdf.stem.replace(" ", "_").replace("-", "_")
            out_filename = f"PDF_{safe_name}.md"
            
            md_content = f"""# {pdf.stem}

**Source:** PDF Document ({pdf.name})

---

{full_text}
"""
            # Save
            out_path = cleaned_dir / out_filename
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(md_content)
                
            print(f"[OK] Exported: {out_filename}")
            count += 1
            
        except Exception as e:
            print(f"[ERROR] Failed to process {pdf.name}: {e}")
            
    print(f"\n[DONE] Successfully exported {count} PDFs to {cleaned_dir}")

if __name__ == "__main__":
    export_pdfs_to_md()
