"""
CollegeWeb AI - Knowledge Indexing Module
Converts content into vector embeddings and stores in FAISS.
Persistent storage implementation.
"""
import os
import json
import pickle
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Import text cleaner with repetition-based noise detection
from modules.text_cleaner import clean_webpage_text, build_noise_lines, clean_markdown

# Import CSE department keywords for chunk classification
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CSE_DEPT_KEYWORDS

# Storage paths
STORAGE_DIR = Path(__file__).parent.parent / "storage"
STORAGE_DIR.mkdir(exist_ok=True)


@dataclass
class KnowledgeItem:
    """Represents a piece of knowledge in the index."""
    id: int
    content: str
    source_type: str  # 'webpage' or 'pdf'
    source_url: str   # URL or file path
    source_name: str  # Page title or PDF name
    doc_type: str     # Document category
    page_number: Optional[int]  # For PDFs
    metadata: Dict


class KnowledgeIndex:
    """
    Vector-based knowledge index using FAISS and Sentence Transformers.
    All data is persisted to disk for survival across restarts.
    """
    
    
    def __init__(self):
        # Model is lazy-loaded on first use to keep startup fast
        self._model = None
        self._model_name = "BAAI/bge-base-en-v1.5"
        self._model_lock = threading.Lock()  # thread-safe lazy init
        self.dimension = 768  # BGE-base uses 768 dimensions
        self.batch_size = 32
        
        # Improved chunking parameters
        self.chunk_size = 500
        self.chunk_overlap = 80
        
        # Storage paths
        self.index_path = STORAGE_DIR / "knowledge.faiss"
        self.meta_path = STORAGE_DIR / "metadata.pkl"
        self.status_path = STORAGE_DIR / "index_status.json"

        # Knowledge items storage
        self.items: List[KnowledgeItem] = []
        self.index = None

        self.load_status()
        self.load_index()   # Only loads FAISS from disk — fast

    @property
    def model(self):
        """Thread-safe lazy-load of the embedding model on first use."""
        if self._model is None:                        # fast path (no lock)
            with self._model_lock:                     # only one thread loads
                if self._model is None:                # double-checked locking
                    print(f"[EMBED] ⏳ Initializing embedding model… (one-time setup)")
                    self._model = SentenceTransformer(self._model_name)
                    print(f"[EMBED] ✅ Embedding model ready ({self._model_name})")
        return self._model
    
    def load_status(self):
        if not self.status_path.exists():
            self.save_status(False, "", 0, 0, 0)
        try:
            with open(self.status_path) as f:
                self.status = json.load(f)
        except Exception as e:
            print(f"Error loading status: {e}")
            self.save_status(False, "", 0, 0, 0)

    def save_status(self, indexed, url, pages, pdfs, chunks):
        self.status = {
            "indexed": indexed,
            "url": url,
            "pages": pages,
            "pdfs": pdfs,
            "chunks": chunks
        }
        with open(self.status_path, "w") as f:
            json.dump(self.status, f, indent=2)
    
    def load_index(self):
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                with open(self.meta_path, "rb") as f:
                    self.items = pickle.load(f)
                return True
            except Exception as e:
                print(f"[ERROR] Failed to load index (corrupted?): {e}")
                print("[INFO] Starting with empty index...")
                self.index = None
                self.items = []
                return False
        else:
            self.index = None
            self.items = []
            return False

    def save_index(self):
        try:
            faiss.write_index(self.index, str(self.index_path))
            with open(self.meta_path, "wb") as f:
                pickle.dump(self.items, f)
            print(f"[OK] Index saved: {len(self.items)} items")
        except Exception as e:
            print(f"[ERROR] Failed to save index: {e}")
            raise  # Re-raise to let caller know saving failed
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        # Initialize index if not exists when adding content
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
            
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.astype('float32')
    
    # ========================================================
    # TOPIC KEYWORDS for metadata tagging
    # ========================================================
    TOPIC_KEYWORDS = {
        'admission': ['admission', 'eligibility', 'eamcet', 'ecet', 'entrance', 'apply'],
        'fee': ['fee', 'tuition', 'payment', 'scholarship', 'cost'],
        'hostel': ['hostel', 'accommodation', 'mess', 'room', 'dormitory'],
        'placement': ['placement', 'package', 'salary', 'recruit', 'campus'],
        'faculty': ['faculty', 'professor', 'hod', 'dean', 'staff'],
        'course': ['programme', 'course', 'syllabus', 'curriculum', 'branch'],
        'exam': ['exam', 'result', 'grade', 'mark', 'semester'],
        'facilities': ['lab', 'library', 'wifi', 'infrastructure', 'gym'],
    }
    
    def _detect_topic(self, content: str, fallback: str = "general") -> str:
        """Detect topic from content using keyword matching."""
        content_lower = content.lower()
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            if any(kw in content_lower for kw in keywords):
                return topic
        return fallback
    
    def _classify_dept(self, content: str, source_name: str = "", section_title: str = "") -> str:
        """Classify a chunk as 'CSE' or 'OTHER' based on keyword matching.
        
        Checks content, source filename, and section title against CSE_DEPT_KEYWORDS.
        """
        combined = f"{content} {source_name} {section_title}".lower()
        if any(kw in combined for kw in CSE_DEPT_KEYWORDS):
            return 'CSE'
        return 'OTHER'
    
    def _topic_based_chunk(self, text: str, source_name: str) -> List[Dict]:
        """
        Split markdown by ## headings to create topic-focused chunks.
        
        Pipeline: .md file → split by headings → each section = 1 chunk
        
        Falls back to paragraph chunking if no headings found.
        """
        import re
        
        # Split by markdown headings (# ## ###)
        # Keep the heading with its content
        sections = re.split(r'\n(?=#{1,3}\s+)', text)
        
        chunks = []
        for section in sections:
            section = section.strip()
            if not section or len(section) < 30:
                continue
            
            # Extract heading as topic title
            heading_match = re.match(r'^(#{1,3})\s+(.+?)$', section, re.MULTILINE)
            if heading_match:
                topic_title = heading_match.group(2).strip()
            else:
                topic_title = source_name
            
            # Detect semantic topic
            detected_topic = self._detect_topic(section, topic_title)
            
            # If section is too long (>1500 chars), sub-chunk by paragraphs
            if len(section) > 1500:
                sub_chunks = self._recursive_chunk_text(section, section_title=topic_title)
                for sub in sub_chunks:
                    sub['topic'] = detected_topic
                    chunks.append(sub)
            else:
                chunks.append({
                    'content': section,
                    'section': topic_title,
                    'topic': detected_topic
                })
        
        # Fallback: if no heading-based sections, use paragraph chunking
        if not chunks:
            print(f"[CHUNK] No headings found in {source_name}, using paragraph chunking")
            return self._recursive_chunk_text(text, section_title=source_name)
        
        print(f"[CHUNK] Split {source_name} into {len(chunks)} topic chunks")
        return chunks
    
    def _recursive_chunk_text(self, text: str, section_title: str = "") -> List[Dict]:
        """
        Recursively split text into optimal chunks.
        
        Strategy:
        1. Split by paragraphs first (\n\n)
        2. If paragraph too long, split by sentences
        3. Apply overlap for context preservation
        4. Preserve section titles in metadata
        """
        import re
        
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_section = section_title
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if this is a section header (starts with common patterns)
            header_match = re.match(r'^#+\s*(.+)$|^([A-Z][^.!?]*):$|^\*\*(.+)\*\*$', para)
            if header_match:
                # Update section title
                current_section = header_match.group(1) or header_match.group(2) or header_match.group(3) or section_title
            
            # Check if adding this paragraph exceeds chunk size
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                # Save current chunk if not empty
                if current_chunk.strip():
                    chunks.append({
                        'content': current_chunk.strip(),
                        'section': current_section
                    })
                
                # If paragraph itself is too long, split by sentences
                if len(para) > self.chunk_size:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    sent_chunk = ""
                    for sent in sentences:
                        if len(sent_chunk) + len(sent) <= self.chunk_size:
                            sent_chunk += sent + " "
                        else:
                            if sent_chunk.strip():
                                chunks.append({
                                    'content': sent_chunk.strip(),
                                    'section': current_section
                                })
                            sent_chunk = sent + " "
                    if sent_chunk.strip():
                        current_chunk = sent_chunk
                    else:
                        current_chunk = ""
                else:
                    current_chunk = para + "\n\n"
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'section': current_section
            })
        
        # Apply overlap between chunks
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            content = chunk['content']
            
            # Add overlap from previous chunk
            if i > 0 and self.chunk_overlap > 0:
                prev_content = chunks[i-1]['content']
                overlap_text = prev_content[-self.chunk_overlap:] if len(prev_content) > self.chunk_overlap else prev_content
                content = "..." + overlap_text + " " + content
            
            overlapped_chunks.append({
                'content': content,
                'section': chunk['section']
            })
        
        return overlapped_chunks
    
    def add_webpage_content(self, pages: List[Dict], skip_cleaning: bool = False) -> int:
        """Add scraped webpage content to the index with improved chunking.
        
        Args:
            pages: List of page dicts with 'markdown' or 'clean_text', 'title', 'url'
            skip_cleaning: If True, skip noise detection and text cleaning (use for pre-cleaned content)
        """
        items_added = 0
        texts_to_encode = []
        items_to_add = []
        
        print(f"[*] Indexing {len(pages)} webpages with BGE embeddings...")
        
        # ========================================================
        # STEP 1: Build noise lines using REPETITION-BASED detection
        # Lines appearing in many pages are boilerplate (nav, footer, etc)
        # SKIP if content is already cleaned
        # ========================================================
        noise_lines = set()
        if not skip_cleaning:
            print(f"[*] Analyzing {len(pages)} pages for repeated boilerplate lines...")
            noise_lines = build_noise_lines(pages)
            print(f"[*] Found {len(noise_lines)} noise lines (appearing in many pages)")
        else:
            print(f"[*] Skipping noise detection (pre-cleaned content)")
        
        for page in pages:
            # Support both old format (clean_text) and new format (markdown)
            raw_content = page.get('clean_text', '') or page.get('markdown', '')
            
            # ========================================================
            # STEP 2: Clean using repetition-based noise detection FIRST
            # Then apply pattern-based fallback
            # SKIP if content is already cleaned
            # ========================================================
            if skip_cleaning:
                content = raw_content  # Use as-is
            else:
                content = clean_markdown(raw_content, noise_lines)  # Repetition-based
                content = clean_webpage_text(content, noise_lines)  # Pattern fallback
            
            if len(content) < 50:
                continue
            
            title = page.get('title', 'Untitled')
            
            # Use TOPIC-BASED chunking (split by ## headings)
            chunks = self._topic_based_chunk(content, source_name=title)
            
            for idx, chunk_data in enumerate(chunks):
                chunk_content = chunk_data['content']
                section = chunk_data.get('section', title)
                topic = chunk_data.get('topic', 'general')
                
                # Skip very short chunks
                if len(chunk_content) < 30:
                    continue
                
                item = KnowledgeItem(
                    id=len(self.items) + len(items_to_add),
                    content=chunk_content,
                    source_type='webpage',
                    source_url=page.get('url', ''),
                    source_name=title,
                    doc_type='webpage',
                    page_number=None,
                    metadata={
                    'chunk_index': idx,
                    'total_chunks': len(chunks),
                    'section': section,
                    'topic': topic,  # NEW: topic metadata for filtering
                    'scraped_at': page.get('scraped_at', ''),
                    'dept': self._classify_dept(chunk_content, title, section),
                }    )
                items_to_add.append(item)
                texts_to_encode.append(chunk_content)
                items_added += 1
        
        if texts_to_encode:
            print(f"[*] Encoding {len(texts_to_encode)} chunks with {self._model_name}...")
            embeddings = self._encode_texts(texts_to_encode)
            self.index.add(embeddings)
            self.items.extend(items_to_add)
        
        print(f"[OK] Added {items_added} webpage chunks to index")
        return items_added
    
    def add_pdf_chunks(self, chunks: List[Dict]) -> int:
        """Add PDF chunks to the index with TOPIC-BASED chunking."""
        items_added = 0
        texts_to_encode = []
        items_to_add = []
        
        print(f"[*] Indexing {len(chunks)} PDF pages with topic-based chunking...")
        
        for chunk in chunks:
            content = chunk.get('content', '')
            if len(content) < 20:
                continue
            
            pdf_name = chunk.get('pdf_name', 'Unknown PDF')
            page_number = chunk.get('page_number')
            
            # Apply topic-based chunking to PDF content
            topic_chunks = self._topic_based_chunk(content, source_name=pdf_name)
            
            for idx, tc in enumerate(topic_chunks):
                tc_content = tc['content']
                section = tc.get('section', pdf_name)
                topic = tc.get('topic', 'general')
                
                if len(tc_content) < 20:
                    continue
                
                item = KnowledgeItem(
                    id=len(self.items) + len(items_to_add),
                    content=tc_content,
                    source_type='pdf',
                    source_url=chunk.get('pdf_path', ''),
                    source_name=pdf_name,
                    doc_type=chunk.get('doc_type', 'general'),
                    page_number=page_number,
                    metadata={
                    'section': section,
                    'topic': topic,  # NEW: topic metadata
                    'chunk_index': idx,
                    'dept': self._classify_dept(tc_content, pdf_name, section),
                    **chunk.get('metadata', {})
                }    )
                items_to_add.append(item)
                texts_to_encode.append(tc_content)
                items_added += 1
        
        if texts_to_encode:
            embeddings = self._encode_texts(texts_to_encode)
            self.index.add(embeddings)
            self.items.extend(items_to_add)
        
        print(f"[OK] Added {items_added} PDF topic chunks to index")
        return items_added
    
    def add_markdown_file(self, filepath: Path, source_name: str = "Faculty Directory") -> int:
        """Add structured markdown content (like faculty.md) to the index."""
        from pathlib import Path
        
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"[INFO] {filepath} not found, skipping...")
            return 0
        
        items_added = 0
        texts_to_encode = []
        items_to_add = []
        
        print(f"[*] Indexing markdown file: {filepath.name}...")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if len(content) < 50:
                print(f"[SKIP] File too small")
                return 0
            
            # Use TOPIC-BASED chunking for markdown files
            chunks = self._topic_based_chunk(content, source_name=source_name)
            
            for idx, chunk_data in enumerate(chunks):
                chunk_content = chunk_data['content']
                section = chunk_data.get('section', source_name)
                topic = chunk_data.get('topic', 'faculty')
                
                # Skip very short chunks
                if len(chunk_content) < 30:
                    continue
                
                item = KnowledgeItem(
                    id=len(self.items) + len(items_to_add),
                    content=chunk_content,
                    source_type='markdown',
                    source_url=str(filepath),
                    source_name=source_name,
                    doc_type='faculty_directory',
                    page_number=None,
                    metadata={
                    'chunk_index': idx,
                    'total_chunks': len(chunks),
                    'section': section,
                    'topic': topic,  # NEW: topic metadata
                    'file_type': 'markdown',
                    'indexed_at': datetime.now().isoformat(),
                    'dept': self._classify_dept(chunk_content, source_name, section),
                }    )
                items_to_add.append(item)
                texts_to_encode.append(chunk_content)
                items_added += 1
            
            if texts_to_encode:
                print(f"[*] Encoding {len(texts_to_encode)} faculty chunks...")
                embeddings = self._encode_texts(texts_to_encode)
                self.index.add(embeddings)
                self.items.extend(items_to_add)
            
            print(f"[OK] Added {items_added} faculty chunks from {filepath.name}")
            return items_added
            
        except Exception as e:
            print(f"[ERROR] Failed to index {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return 0
    
    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.3) -> List[Tuple[KnowledgeItem, float]]:
        """Search for relevant content using semantic similarity with source-aware filtering."""
        print(f"[INDEX SEARCH] Query: '{query[:50]}...', top_k={top_k}, min_similarity={min_similarity}")
        
        if self.index is None or self.index.ntotal == 0:
            print(f"[INDEX SEARCH ERROR] Index is None or empty! ntotal={self.index.ntotal if self.index else 'None'}")
            return []
        
        print(f"[INDEX SEARCH] Index has {self.index.ntotal} items, encoding query...")
        
        try:
            query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
            print(f"[INDEX SEARCH] Query encoded, searching FAISS...")
            distances, indices = self.index.search(query_embedding, min(top_k * 3, self.index.ntotal))
            print(f"[INDEX SEARCH] FAISS search complete. Processing {len(indices[0])} results...")
        except Exception as e:
            print(f"[INDEX SEARCH ERROR] Encoding or FAISS search failed: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # ========================================================
        # SOURCE-AWARE RETRIEVAL: Smart Filtering
        # ========================================================
        query_lower = query.lower()
        
        # DEFINITIONS
        # 1. Queries that should prefer WEBPAGEs (mostly static info)
        webpage_priority_keywords = [
            "fee", "tuition", "admission", "admissions", "hostel", 
            "contact", "address", "phone", "email", "principal", "about",
            "eligibility", "cut off", "cutoff", "ranking", "rank"
        ]
        
        # 2. Queries that might need PDFs (Placements, Syllabus, Calendars)
        pdf_priority_keywords = [
            "syllabus", "curriculum", "calendar", "handbook", "brochure",
            "placements", "placement report", "regulation",
            "fee", "fees"
        ]

        # 3. Junk PDFs that confuse the LLM (Financial audits, NAAC reports)
        bad_pdf_patterns = ["naac", "audit", "budget", "finance", "expenditure", "balance sheet"]
        
        # 4. High-value PDFs
        good_pdf_patterns = ["placement", "brochure", "syllabus", "calendar", "academic", "handbook", "fees"]
        
        prefer_webpage = any(kw in query_lower for kw in webpage_priority_keywords)
        prefer_faculty = "faculty" in query_lower or "professor" in query_lower
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.items):
                continue
            
            item = self.items[idx]
            similarity = 1 / (1 + dist)
            adjusted_similarity = similarity
            
            # ========================================================
            # SCORING ADJUSTMENTS
            # ========================================================
            
            # 1. Handle PDFs
            if item.source_type == 'pdf':
                pdf_name = item.source_name.lower()
                
                # Penalize "Junk" PDFs (NAAC, Audit, etc) - ALWAYS
                if any(bad in pdf_name for bad in bad_pdf_patterns):
                    adjusted_similarity *= 0.4  # Heavy penalty
                    # print(f"[FILTER] Junk PDF '{item.source_name}' penalized heavily")
                
                # Boost "Good" PDFs (Placements, Syllabus)
                elif any(good in pdf_name for good in good_pdf_patterns):
                    adjusted_similarity *= 1.15
                    # print(f"[FILTER] Good PDF '{item.source_name}' boosted")
                    
                # If query strictly wants webpage (e.g. "fees"), penalize generic PDFs
                elif prefer_webpage:
                    adjusted_similarity *= 0.6
                
                # Otherwise (neutral PDF), slight penalty to prefer content
                else:
                    adjusted_similarity *= 0.9

            # 2. Handle Webpages
            elif item.source_type == 'webpage':
                # Boost if query matches webpage priority topics
                if prefer_webpage:
                    adjusted_similarity *= 1.2
                
                # Penalize very short content (nav menus)
                if len(item.content) < 100:
                    adjusted_similarity *= 0.6
            
            # 3. Handle Faculty Data
            elif item.doc_type == 'markdown' or 'faculty' in item.source_name.lower():
                if prefer_faculty:
                    adjusted_similarity *= 1.4  # Strong boost for faculty queries
            
            # Threshold Check
            if adjusted_similarity >= min_similarity:
                results.append((item, float(adjusted_similarity)))
        
        # Sort results
        results.sort(key=lambda x: x[1], reverse=True)
        
        # ========================================================
        # DYNAMIC CUTOFF
        # ========================================================
        # If we have very high confidence results (>0.7), drop the low confidence ones (<0.5)
        # This reduces noise validation
        if results and results[0][1] > 0.7:
            results = [r for r in results if r[1] > 0.5]
            
        final_results = results[:top_k]
        
        # Log source breakdown
        sources = {}
        for item, score in final_results:
            st = item.source_type
            sources[st] = sources.get(st, 0) + 1
        
        print(f"[SEARCH] Returning {len(final_results)} results. Sources: {sources}")
        return final_results
    
    def search_by_type(self, query: str, doc_type: str, top_k: int = 5) -> List[Tuple[KnowledgeItem, float]]:
        """Search within a specific document type."""
        all_results = self.search(query, top_k * 3)
        filtered = [
            (item, score) for item, score in all_results 
            if item.doc_type == doc_type
        ]
        return filtered[:top_k]
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        doc_types = {}
        source_types = {'webpage': 0, 'pdf': 0}
        
        for item in self.items:
            doc_type = item.doc_type
            if doc_type not in doc_types:
                doc_types[doc_type] = 0
            doc_types[doc_type] += 1
            source_types[item.source_type] += 1
        
        return {
            'total_items': len(self.items),
            'index_size': self.index.ntotal if self.index else 0,
            'by_doc_type': doc_types,
            'by_source_type': source_types,
            'status': self.status
        }
    
    def clear(self):
        """Clear the entire index."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.items = []
        self.save_status(False, "", 0, 0, 0)
        # Remove files
        if self.index_path.exists():
            self.index_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()
        print("[OK] Index cleared")


# Singleton instance
_knowledge_index: Optional[KnowledgeIndex] = None


def get_knowledge_index() -> KnowledgeIndex:
    """Get or create the knowledge index singleton."""
    global _knowledge_index
    if _knowledge_index is None:
        _knowledge_index = KnowledgeIndex()
    return _knowledge_index


if __name__ == "__main__":
    # Test
    index = KnowledgeIndex()
    print(f"Status: {index.status}")
    print(f"Stats: {index.get_stats()}")
