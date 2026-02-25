from pathlib import Path
from config import PDF_DIR, DATA_DIR
import json
from dataclasses import dataclass
from typing import List
from pypdf import PdfReader
import logging

# Suppress pypdf warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)


@dataclass
class PDFChunk:
    content: str
    pdf_path: str
    pdf_name: str
    page_number: int
    doc_type: str
    metadata: dict


class PDFProcessor:

    def __init__(self):
        self.all_chunks: List[PDFChunk] = []

    # Smart text chunking (very important)
    def split_text(self, text, chunk_size=900, overlap=150):
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap
        return chunks

    def process_all_pdfs(self, progress_callback=None):
        """Process all PDF files and extract text chunks."""
        self.all_chunks = []
        pdf_files = list(PDF_DIR.glob("*.pdf"))
        total_pdfs = len(pdf_files)
        print(f"[*] Processing {total_pdfs} PDF files...")

        for idx, pdf_path in enumerate(pdf_files):
            try:
                reader = PdfReader(pdf_path)
                pdf_name = pdf_path.name

                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    if not text:
                        continue

                    for part in self.split_text(text):
                        self.all_chunks.append(PDFChunk(
                            content=part.strip(),
                            pdf_path=str(pdf_path),
                            pdf_name=pdf_name,
                            page_number=page_num,
                            doc_type="pdf",
                            metadata={}
                        ))
                
                # Report progress after each PDF
                if progress_callback:
                    progress_callback({
                        "current": idx + 1,
                        "total": total_pdfs,
                        "file": pdf_name
                    })
                    
            except Exception as e:
                print(f"Skipping broken PDF {pdf_path.name}: {e}")

        return {
            "total_pdfs": total_pdfs,
            "total_chunks": len(self.all_chunks)
        }

    def save_processed_data(self):
        output = {
            "total_pdfs": len({c.pdf_name for c in self.all_chunks}),
            "total_chunks": len(self.all_chunks),
            "pdfs": []
        }

        for pdf_name in set(c.pdf_name for c in self.all_chunks):
            chunks = [c for c in self.all_chunks if c.pdf_name == pdf_name]
            output["pdfs"].append({
                "name": pdf_name,
                "title": pdf_name,
                "doc_type": "pdf",
                "total_pages": len({c.page_number for c in chunks}),
                "total_chunks": len(chunks)
            })

        with open(DATA_DIR / "processed_pdfs.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
