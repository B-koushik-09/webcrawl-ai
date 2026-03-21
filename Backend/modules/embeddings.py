"""
CollegeWeb AI - Knowledge Indexing Module
Converts content into vector embeddings and stores in ChromaDB.
Persistent storage implementation.
"""
import os
import json
import uuid
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import chromadb

# Import text cleaner with repetition-based noise detection
from modules.text_cleaner import clean_webpage_text, build_noise_lines, clean_markdown

# Import CSE department keywords for chunk classification
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CSE_DEPT_KEYWORDS, CSE_SUB_DEPT_KEYWORDS

# Storage paths
STORAGE_DIR = Path(__file__).parent.parent / "storage"
STORAGE_DIR.mkdir(exist_ok=True)


@dataclass
class KnowledgeItem:
    """Represents a piece of knowledge in the index."""
    id: str  # Kept as str for ChromaDB compatibility
    content: str
    source_type: str  # 'webpage' or 'pdf'
    source_url: str   # URL or file path
    source_name: str  # Page title or PDF name
    doc_type: str     # Document category
    page_number: Optional[int]  # For PDFs
    metadata: Dict


class KnowledgeIndex:
    """
    Vector-based knowledge index using ChromaDB and Sentence Transformers.
    All data is persisted to disk natively by ChromaDB.
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
        self.chroma_path = STORAGE_DIR / "chromadb"
        self.status_path = STORAGE_DIR / "index_status.json"
        
        print(f"[INDEX] Initializing ChromaDB at {self.chroma_path}")
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        
        # Use Cosine Similarity for the vector space
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )

        self.load_status()

    @property
    def model(self):
        """Thread-safe lazy-load of the embedding model on first use."""
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    print(f"[EMBED] [*] Initializing embedding model... (one-time setup)")
                    # Lazy import: keeps startup fast by deferring the heavy PyTorch import
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self._model_name)
                    print(f"[EMBED] [OK] Embedding model ready ({self._model_name})")
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
        """Kept for backward compatibility. ChromaDB loads automatically."""
        return True

    def save_index(self):
        """Kept for backward compatibility. ChromaDB persists automatically."""
        pass
    
    def _encode_texts(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into embeddings."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        # ChromaDB expects Python lists
        return embeddings.tolist()
    
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
        """Classify a chunk as 'CSE', 'CSE-DS', 'CSE-CYS', 'CSE-AIML', 'CSE-IOT', or 'OTHER'."""
        combined = f"{content} {source_name} {section_title}".lower()
        # Check sub-department keywords FIRST (more specific matches)
        for sub_dept_tag, keywords in CSE_SUB_DEPT_KEYWORDS.items():
            if any(kw in combined for kw in keywords):
                return sub_dept_tag
        # Then check generic CSE
        if any(kw in combined for kw in CSE_DEPT_KEYWORDS):
            return 'CSE'
        return 'OTHER'
    
    def _topic_based_chunk(self, text: str, source_name: str) -> List[Dict]:
        """
        Split markdown by ## headings to create topic-focused chunks.
        """
        import re
        sections = re.split(r'\n(?=#{1,3}\s+)', text)
        chunks: List[Dict] = []
        for section in sections:
            section = section.strip()
            if not section or len(section) < 30:
                continue
            
            heading_match = re.match(r'^(#{1,3})\s+(.+?)$', section, re.MULTILINE)
            if heading_match:
                topic_title = heading_match.group(2).strip()
            else:
                topic_title = source_name
            
            detected_topic = self._detect_topic(section, topic_title)
            
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
        
        if not chunks:
            print(f"[CHUNK] No headings found in {source_name}, using paragraph chunking")
            return self._recursive_chunk_text(text, section_title=source_name)
        
        return chunks
    
    def _recursive_chunk_text(self, text: str, section_title: str = "") -> List[Dict]:
        """Recursively split text into optimal chunks."""
        import re
        chunks: List[Dict] = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        current_section = section_title
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            header_match = re.match(r'^#+\s*(.+)$|^([A-Z][^.!?]*):$|^\*\*(.+)\*\*$', para)
            if header_match:
                current_section = header_match.group(1) or header_match.group(2) or header_match.group(3) or section_title
            
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append({
                        'content': current_chunk.strip(),
                        'section': current_section
                    })
                
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
        
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'section': current_section
            })
        
        overlapped_chunks: List[Dict] = []
        for i, chunk in enumerate(chunks):
            content = chunk['content']
            if i > 0 and self.chunk_overlap > 0:
                prev_content = chunks[i-1]['content']
                overlap_text = prev_content[-self.chunk_overlap:] if len(prev_content) > self.chunk_overlap else prev_content
                content = "..." + overlap_text + " " + content
            
            overlapped_chunks.append({
                'content': content,
                'section': chunk['section']
            })
        
        return overlapped_chunks

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """ChromaDB metadata must only have string, int, float, or bool values."""
        clean: Dict = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if type(v) in [str, int, float, bool]:
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    def _add_to_chroma(self, items: List[KnowledgeItem]):
        """Helper to batch add items to ChromaDB."""
        if not items:
            return
        
        texts = [i.content for i in items]
        ids = [i.id for i in items]
        metadatas: List[Dict] = []
        for i in items:
            meta = {
                "source_type": i.source_type,
                "source_url": i.source_url,
                "source_name": i.source_name,
                "doc_type": i.doc_type,
            }
            if i.page_number is not None:
                meta["page_number"] = i.page_number
            meta.update(i.metadata)
            metadatas.append(self._sanitize_metadata(meta))
            
        embeddings = self._encode_texts(texts)
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    
    def add_webpage_content(self, pages: List[Dict], skip_cleaning: bool = False) -> int:
        """Add scraped webpage content to the index."""
        items_added: int = 0
        items_to_add = []
        
        print(f"[*] Indexing {len(pages)} webpages with ChromaDB...")
        
        noise_lines: set = set()
        if not skip_cleaning:
            print(f"[*] Analyzing {len(pages)} pages for repeated boilerplate lines...")
            noise_lines = build_noise_lines(pages)
        
        for page in pages:
            raw_content = page.get('clean_text', '') or page.get('markdown', '')
            if skip_cleaning:
                content = raw_content
            else:
                content = clean_markdown(raw_content, noise_lines)
                content = clean_webpage_text(content, noise_lines)
            
            if len(content) < 50:
                continue
            
            title = page.get('title', 'Untitled')
            chunks = self._topic_based_chunk(content, source_name=title)
            
            for idx, chunk_data in enumerate(chunks):
                chunk_content = chunk_data['content']
                if len(chunk_content) < 30:
                    continue
                
                # We do some quick role checking here
                content_lower = chunk_content.lower()
                is_leadership = any(role in content_lower for role in ['chairman', 'director', 'principal', 'dean'])
                
                item = KnowledgeItem(
                    id=f"web_{uuid.uuid4().hex[:16]}",
                    content=chunk_content,
                    source_type='webpage',
                    source_url=page.get('url', ''),
                    source_name=title,
                    doc_type='webpage',
                    page_number=None,
                    metadata={
                        'chunk_index': idx,
                        'total_chunks': len(chunks),
                        'section': chunk_data.get('section', title),
                        'topic': chunk_data.get('topic', 'general'),
                        'scraped_at': page.get('scraped_at', ''),
                        'dept': self._classify_dept(chunk_content, title, chunk_data.get('section', title)),
                        'is_leadership': is_leadership
                    }
                )
                items_to_add.append(item)
                items_added = items_added + 1  # pyre-ignore[58]
                if len(items_to_add) >= 100:
                    self._add_to_chroma(items_to_add)
                    items_to_add.clear()
        
        if items_to_add:
            self._add_to_chroma(items_to_add)
            
        print(f"[OK] Added {items_added} webpage chunks to index")
        return items_added
    
    def add_pdf_chunks(self, chunks: List[Dict]) -> int:
        """Add PDF chunks to the index."""
        items_added: int = 0
        items_to_add: List[KnowledgeItem] = []
        
        print(f"[*] Indexing {len(chunks)} PDF pages with ChromaDB...")
        
        for chunk in chunks:
            content = chunk.get('content', '')
            if len(content) < 20:
                continue
            
            pdf_name = chunk.get('pdf_name', 'Unknown PDF')
            page_number = chunk.get('page_number')
            topic_chunks = self._topic_based_chunk(content, source_name=pdf_name)
            
            for idx, tc in enumerate(topic_chunks):
                tc_content = tc['content']
                if len(tc_content) < 20:
                    continue
                
                item = KnowledgeItem(
                    id=f"pdf_{uuid.uuid4().hex[:16]}",
                    content=tc_content,
                    source_type='pdf',
                    source_url=chunk.get('pdf_path', ''),
                    source_name=pdf_name,
                    doc_type=chunk.get('doc_type', 'pdf'),
                    page_number=page_number,
                    metadata={
                        'section': tc.get('section', pdf_name),
                        'topic': tc.get('topic', 'general'),
                        'chunk_index': idx,
                        'dept': self._classify_dept(tc_content, pdf_name, tc.get('section', pdf_name)),
                    }
                )
                # Overwrite metadata from caller
                for k, v in chunk.get('metadata', {}).items():
                    item.metadata[k] = v
                items_to_add.append(item)
                items_added = items_added + 1  # pyre-ignore[58]
                
                if len(items_to_add) >= 100:
                    self._add_to_chroma(items_to_add)
                    items_to_add.clear()
                    
        if items_to_add:
            self._add_to_chroma(items_to_add)
            
        print(f"[OK] Added {items_added} PDF topic chunks to index")
        return items_added
    
    def add_markdown_file(self, filepath: Path, source_name: str = "Faculty Directory") -> int:
        """Add structured markdown content (like faculty.md) to the index."""
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"[INFO] {filepath} not found, skipping...")
            return 0
        
        print(f"[*] Indexing markdown file: {filepath.name}...")
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            if len(content) < 50:
                return 0
            
            chunks = self._topic_based_chunk(content, source_name=source_name)
            items_to_add: List[KnowledgeItem] = []
            
            for idx, chunk_data in enumerate(chunks):
                chunk_content = chunk_data['content']
                if len(chunk_content) < 30:
                    continue
                
                content_lower = chunk_content.lower()
                role = "faculty"
                if "hod" in content_lower or "head" in content_lower:
                    role = "hod"
                elif "principal" in content_lower:
                    role = "principal"
                elif "chairman" in content_lower or "chairperson" in content_lower:
                    role = "chairman"
                elif "director" in content_lower:
                    role = "director"
                elif "dean" in content_lower:
                    role = "dean"
                    
                is_leadership = role != "faculty"
                
                item = KnowledgeItem(
                    id=f"md_{uuid.uuid4().hex[:16]}",
                    content=chunk_content,
                    source_type='markdown',
                    source_url=str(filepath),
                    source_name=source_name,
                    doc_type='faculty_directory',
                    page_number=None,
                    metadata={
                        'chunk_index': idx,
                        'total_chunks': len(chunks),
                        'section': chunk_data.get('section', source_name),
                        'topic': chunk_data.get('topic', 'faculty'),
                        'file_type': 'markdown',
                        'indexed_at': datetime.now().isoformat(),
                        'dept': self._classify_dept(chunk_content, source_name, chunk_data.get('section', source_name)),
                        'role': role,
                        'is_leadership': is_leadership
                    }
                )
                items_to_add.append(item)
            
            if items_to_add:
                self._add_to_chroma(items_to_add)
            print(f"[OK] Added {len(items_to_add)} markdown chunks to index")
            return len(items_to_add)
        except Exception as e:
            print(f"[ERROR] Failed to index {filepath}: {e}")
            return 0

    def search(self, query: str, top_k: int = 5, min_similarity: float = 0.3, where: Optional[Dict] = None) -> List[Tuple[KnowledgeItem, float]]:
        """Search ChromaDB using metadata filters."""
        print(f"[INDEX SEARCH] Query: '{query[:50]}...', top_k={top_k}, where={where}")
        
        try:
            query_embedding = self._encode_texts([query])[0]
            
            # Use chromadb's query
            # We request more results initially if min_similarity clipping is strict
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2, 
                where=where
            )
            
            if not results["ids"] or not results["ids"][0]:
                print("[INDEX SEARCH] No results found in ChromaDB.")
                return []
                
            final_results: List[Tuple[KnowledgeItem, float]] = []
            
            # ChromaDB cosine returns distance. Similarity = 1 - distance
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                content = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                
                similarity = max(0.0, 1.0 - distance)
                
                if similarity >= min_similarity:
                    item = KnowledgeItem(
                        id=doc_id,
                        content=content,
                        source_type=metadata.get("source_type", "unknown"),
                        source_url=metadata.get("source_url", ""),
                        source_name=metadata.get("source_name", ""),
                        doc_type=metadata.get("doc_type", "unknown"),
                        page_number=int(metadata.get("page_number")) if "page_number" in metadata else None,
                        metadata=metadata
                    )
                    final_results.append((item, float(similarity)))
                    
            # Sort by similarity
            final_results.sort(key=lambda x: x[1], reverse=True)
            
            # Dynamic cutoff logic: drop noisy low-confidence tails if top results are great
            if len(final_results) > 0:
                first_item = final_results[0]  # pyre-ignore[16]
                if first_item[1] > 0.7:
                    final_results = [r for r in final_results if r[1] > 0.5]
                
            return final_results[:top_k]  # pyre-ignore
            
        except Exception as e:
            print(f"[INDEX SEARCH ERROR]: {e}")
            import traceback
            traceback.print_exc()
            return []

    def search_by_type(self, query: str, doc_type: str, top_k: int = 5) -> List[Tuple[KnowledgeItem, float]]:
        """Search within a specific document type."""
        return self.search(query, top_k=top_k, where={"doc_type": doc_type})

    def get_stats(self) -> Dict:
        """Get index statistics."""
        try:
            count = self.collection.count()
        except Exception:
            count = 0
            
        return {
            'total_items': count,
            'index_size': count,
            'status': self.status
        }
    
    def clear(self):
        """Clear the entire index."""
        try:
            self.client.delete_collection("knowledge_base")
            self.collection = self.client.create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
            self.save_status(False, "", 0, 0, 0)
            print("[OK] Index cleared")
        except Exception as e:
            print(f"Error clearing: {e}")


# Singleton instance
_knowledge_index: Optional[KnowledgeIndex] = None

def get_knowledge_index() -> KnowledgeIndex:
    """Get or create the knowledge index singleton."""
    global _knowledge_index
    if _knowledge_index is None:
        _knowledge_index = KnowledgeIndex()
    return _knowledge_index


if __name__ == "__main__":
    index = KnowledgeIndex()
    print(f"Stats: {index.get_stats()}")
