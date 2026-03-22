"""
VNRVJIET AI - Main FastAPI Application
Intelligent College Website Knowledge Assistant
"""
import asyncio
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(override=True)

import os
key = os.getenv('FIRECRAWL_API_KEY')
if key:
    print(f"\n[STARTUP CHECK] [OK] Firecrawl API Key loaded: ...{key[-4:]}")
else:
    print("\n[STARTUP CHECK] [X] Firecrawl API Key MISSING in .env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Import modules
from config import DATA_DIR, VECTOR_STORE_DIR, DEFAULT_TARGET_URL
from modules.embeddings import KnowledgeIndex
from modules.rag_engine import RAGEngine, get_rag_engine, initialize_rag_engine
from modules.admin import AdminManager, get_admin_manager, IndexStatus


# Pydantic models for API
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    query: str = Field(..., min_length=1, max_length=1000, description="User's question")
    doc_type: Optional[str] = Field(None, description="Optional document type filter")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    query: str
    answer: str
    citations: list
    confidence: float
    grounded: bool
    sources_count: int


class StatusResponse(BaseModel):
    """Response model for status endpoint."""
    status: str
    stage: str
    current: int
    total: int
    message: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""
    has_index: bool
    pages_scraped: int
    pdfs_processed: int
    total_chunks: int
    last_updated: Optional[str]
    target_url: Optional[str]


# Global instances
knowledge_index: Optional[KnowledgeIndex] = None
rag_engine: Optional[RAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - loads index on startup."""
    global knowledge_index, rag_engine
    
    print("[*] Starting VNRVJIET AI...")
    
    # Load persistent knowledge index (automatically loads from disk)
    knowledge_index = KnowledgeIndex()
    
    # Check if index has data (from status file OR live ChromaDB collection)
    chunk_count = knowledge_index.collection.count()
    if knowledge_index.status.get('indexed', False) or chunk_count > 0:
        rag_engine = initialize_rag_engine(knowledge_index)
        print(f"[OK] Knowledge index loaded: {chunk_count} chunks in ChromaDB")
    else:
        print("[!] No indexed data found. Please run indexing first.")
        rag_engine = RAGEngine(knowledge_index)
    
    yield
    
    print("[*] Shutting down VNRVJIET AI...")


# Create FastAPI app
app = FastAPI(
    title="VNRVJIET AI",
    description="Intelligent College Website Knowledge Assistant - AI-powered chatbot for college information",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== API Routes ==============

@app.get("/")
async def root():
    """Root endpoint - API health check."""
    return {
        "status": "running",
        "service": "VNRVJIET AI",
        "version": "1.0.0",
        "message": "Intelligent College Website Knowledge Assistant"
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    admin = get_admin_manager()
    stats = admin.get_index_stats()
    
    return {
        "status": "healthy",
        "index_ready": stats['has_index'],
        "pages_indexed": stats['pages_scraped'],
        "pdfs_indexed": stats['pdfs_processed']
    }


# ============== Chat Endpoints ==============

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - Answer questions using the knowledge base.
    
    This endpoint receives a natural language question and returns
    a grounded answer with citations from the college website/documents.
    """
    global rag_engine
    
    print(f"\n{'='*60}")
    print(f"[CHAT REQUEST] Query: {request.query[:100]}")
    print(f"[CHAT REQUEST] Doc Type: {request.doc_type}")
    print(f"{'='*60}")
    
    # Check if RAG engine exists
    if not rag_engine:
        print("[ERROR] RAG engine is None!")
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not initialized. Please run indexing first."
        )
    
    # Check if index exists
    if not rag_engine.index:
        print("[ERROR] RAG engine index is None!")
        raise HTTPException(
            status_code=503,
            detail="Knowledge base index not loaded. Please run indexing or reload index."
        )
    
    # Check if index has data
    if rag_engine.index.collection is None or rag_engine.index.collection.count() == 0:
        print(f"[ERROR] Index is empty! chunk_count = {rag_engine.index.collection.count() if rag_engine.index.collection else '0'}")
        raise HTTPException(
            status_code=503,
            detail="Knowledge base is empty. Please run indexing first."
        )
    
    print(f"[OK] RAG engine ready. Index has {rag_engine.index.collection.count()} items")
    
    try:
        # Query the RAG engine
        print(f"[*] Querying RAG engine...")
        if request.doc_type:
            response = rag_engine.query_with_type_filter(
                request.query, 
                request.doc_type
            )
        else:
            response = rag_engine.query(request.query)
        
        print(f"[OK] Query completed. Confidence: {response.confidence:.2f}")
        
        # Format response
        result = rag_engine.format_response_with_citations(response)
        
        print(f"[OK] Response formatted. Returning {len(result['citations'])} citations")
        return ChatResponse(**result)
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Chat endpoint failed!")
        print(f"[ERROR] Exception type: {type(e).__name__}")
        print(f"[ERROR] Exception message: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")



@app.get("/api/chat/suggestions")
async def get_suggestions():
    """Get suggested questions for the chat interface."""
    return {
        "suggestions": [
            "Who is the principal of VNRVJIET?",
            "Who is the HOD of CSE department?",
            "Who is the HOD of EEE?",
            "What is the fee structure for B.Tech?",
            "What is the highest placement package in CSE?",
            "How many students were placed in Amazon?",
            "What is the average package of CSE?",
            "What are the placement statistics of CSE 2025?",
        ]
    }


# ============== Admin Endpoints ==============

@app.get("/api/admin/status")
async def get_index_status():
    """Get current indexing status."""
    admin = get_admin_manager()
    status = admin.get_status()
    
    # Merge with persistent knowledge status
    if knowledge_index:
        status.update(knowledge_index.status)
        
    return status


@app.get("/api/admin/stats")
async def get_index_stats():
    """Get statistics about the current index."""
    # Prioritize valid knowledge index stats
    if knowledge_index and knowledge_index.status.get("indexed"):
        s = knowledge_index.status
        return {
            "has_index": True,
            "pages_scraped": s.get("pages", 0),
            "pdfs_processed": s.get("pdfs", 0),
            "total_chunks": s.get("chunks", 0),
            "last_updated": s.get("last_updated"),
            "target_url": s.get("url")
        }
        
    # Fallback to admin manager stats (file based)
    admin = get_admin_manager()
    stats = admin.get_index_stats()
    return stats


@app.get("/api/admin/pdfs")
async def get_pdf_list():
    """Get list of indexed PDF documents."""
    admin = get_admin_manager()
    pdfs = admin.get_pdf_list()
    return {"pdfs": pdfs, "total": len(pdfs)}





@app.post("/api/admin/reload")
async def reload_index():
    """Reload the knowledge index from disk."""
    global knowledge_index, rag_engine
    
    try:
        index = KnowledgeIndex()
        if index.load_index():
            knowledge_index = index
            rag_engine = initialize_rag_engine(index)
            stats = index.get_stats()
            return {
                "success": True,
                "message": "Index reloaded successfully",
                "stats": stats
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="No saved index found"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/rebuild")
async def rebuild_index():
    """
    Rebuild the ChromaDB index from existing scraped_pages.json WITHOUT re-crawling.
    
    This is useful when you want to:
    - Apply text cleaning improvements
    - Change embedding parameters
    - Fix index corruption
    
    WITHOUT downloading data again.
    """
    global knowledge_index, rag_engine
    
    admin = get_admin_manager()
    
    # Check if already running
    if admin.current_status in [IndexStatus.SCRAPING, IndexStatus.PROCESSING_PDFS, IndexStatus.INDEXING]:
        raise HTTPException(
            status_code=409,
            detail="Indexing already in progress"
        )
    
    async def run_rebuild():
        global knowledge_index, rag_engine
        
        try:
            print("[*] Starting index rebuild from cleaned_pages/ folder...")
            admin.current_status = IndexStatus.INDEXING
            admin.update_progress('rebuilding', 0, 100, 'Scanning cleaned_pages folder...')
            
            # Create new index
            index = KnowledgeIndex()
            index.clear()  # Clear existing index
            
            # Load cleaned pages from cleaned_pages/ folder
            from config import BASE_DIR, DATA_DIR
            cleaned_dir = BASE_DIR / 'cleaned_pages'
            
            if not cleaned_dir.exists():
                raise FileNotFoundError("cleaned_pages/ folder not found!")
            
            # Get all markdown files
            all_md_files = list(cleaned_dir.glob('*.md'))
            print(f"[*] Found {len(all_md_files)} total markdown files")
            
            # Categorize files:
            # 1. Numbered webpages (001_*.md to 999_*.md) 
            # 2. PDFs (PDF_*.md)
            # 3. Custom files (everything else: 000_contact_info.md, faq.md, INDEX.md, faculty.md)
            
            import re
            webpage_files = []
            pdf_files = []
            custom_files = []
            
            for f in all_md_files:
                if f.name.startswith('PDF_'):
                    pdf_files.append(f)
                elif re.match(r'^\d{3}_', f.name):
                    # Numbered webpage files (001_xxx.md, 134_admission.md, etc.)
                    webpage_files.append(f)
                else:
                    # Custom files: 000_contact_info.md, faq.md, INDEX.md, faculty.md
                    custom_files.append(f)
            
            # Sort numbered files by number
            webpage_files.sort(key=lambda x: x.name)
            
            total_webpages = len(webpage_files)
            total_pdfs = len(pdf_files)
            total_custom = len(custom_files)
            
            print(f"[*] Categorized: {total_webpages} webpages, {total_pdfs} PDFs, {total_custom} custom files")
            
            # ========================================
            # STEP 1: Index numbered webpage files
            # ========================================
            admin.update_progress('rebuilding', 10, 100, f'Indexing {total_webpages} webpages...')
            
            if webpage_files:
                webpage_pages = []
                for md_file in webpage_files:
                    try:
                        with open(md_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if len(content) < 50:
                            continue
                        
                        # Extract title from markdown content or filename
                        title = "VNRVJIET"
                        lines = content.split('\n')
                        for line in lines[:15]:
                            if line.startswith('# '):
                                title = line[2:].strip()
                                break
                            elif '**Source:**' in line:
                                import re
                                match = re.search(r'\[([^\]]+)\]', line)
                                if match:
                                    title = match.group(1)
                                break
                        
                        # Extract URL if present
                        url = "https://vnrvjiet.ac.in/"
                        for line in lines[:15]:
                            if '**Source:**' in line:
                                url_match = re.search(r'\(([^)]+)\)', line)
                                if url_match:
                                    url = url_match.group(1)
                                break
                        
                        webpage_pages.append({
                            'title': title,
                            'url': url,
                            'markdown': content,
                            'scraped_at': ''
                        })
                    except Exception as e:
                        print(f"[WARN] Failed to read {md_file.name}: {e}")
                
                if webpage_pages:
                    print(f"[*] Indexing {len(webpage_pages)} webpages with BGE embeddings...")
                    index.add_webpage_content(webpage_pages, skip_cleaning=True)
            
            # ========================================
            # STEP 2: Index custom markdown files
            # ========================================
            admin.update_progress('rebuilding', 40, 100, f'Indexing {total_custom} custom files...')
            
            if custom_files:
                custom_pages = []
                for md_file in custom_files:
                    try:
                        with open(md_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if len(content) < 50:
                            continue
                        
                        # Extract title
                        title = md_file.stem.replace('_', ' ').title()
                        lines = content.split('\n')
                        for line in lines[:10]:
                            if line.startswith('# '):
                                title = line[2:].strip()
                                break
                        
                        custom_pages.append({
                            'title': title,
                            'url': f"custom/{md_file.name}",
                            'markdown': content,
                            'scraped_at': ''
                        })
                        print(f"[*] Prepared custom file: {md_file.name}")
                    except Exception as e:
                        print(f"[WARN] Failed to read {md_file.name}: {e}")
                
                if custom_pages:
                    print(f"[*] Indexing {len(custom_pages)} custom markdown files...")
                    index.add_webpage_content(custom_pages, skip_cleaning=True)
            
            # ========================================
            # STEP 3: Index PDF markdown files
            # ========================================
            admin.update_progress('rebuilding', 60, 100, f'Indexing {total_pdfs} PDFs...')
            
            if pdf_files:
                print(f"[*] Found {len(pdf_files)} processed PDF markdown files")
                pdf_inputs = []
                
                for pdf_file in pdf_files:
                    try:
                        with open(pdf_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if len(content) < 50: 
                            continue
                            
                        # Clean title (remove PDF_ prefix and extension)
                        title = pdf_file.stem
                        if title.startswith('PDF_'):
                            title = title[4:]
                            
                        pdf_inputs.append({
                            'content': content,
                            'pdf_name': title,
                            'pdf_path': str(pdf_file.name).replace('.md', '.pdf'),
                            'doc_type': 'pdf',
                            'page_number': 1,
                            'metadata': {'source': 'rebuild_from_md'}
                        })
                    except Exception as e:
                        print(f"[WARN] Failed to read {pdf_file.name}: {e}")
                
                if pdf_inputs:
                    print(f"[*] Indexing {len(pdf_inputs)} PDF pages with topic-based chunking...")
                    index.add_pdf_chunks(pdf_inputs)
            
            # ========================================
            # STEP 4: Save index
            # ========================================
            admin.update_progress('rebuilding', 90, 100, 'Saving index...')
            index.save_index()
            
            # Update status
            page_count = total_webpages + total_custom
            pdf_count = total_pdfs
            chunk_count = index.collection.count() if index.collection else 0
            
            # Save persistent status with CORRECT counts
            index.save_status(True, "rebuilt", page_count, pdf_count, chunk_count)
            
            # Reload into memory
            knowledge_index = index
            rag_engine = initialize_rag_engine(index)
            
            admin.current_status = IndexStatus.COMPLETE
            admin.update_progress('complete', 100, 100, f'Index rebuilt: {chunk_count} chunks from {page_count} pages and {pdf_count} PDFs')
            
            print(f"[OK] Index rebuilt: {chunk_count} chunks from {page_count} pages and {pdf_count} PDFs")
            
        except Exception as e:
            print(f"[ERROR] Rebuild failed: {e}")
            import traceback
            traceback.print_exc()
            admin.current_status = IndexStatus.ERROR
            admin.update_progress('error', 0, 0, f'Rebuild failed: {str(e)}')
    
    # Run in background
    asyncio.create_task(run_rebuild())
    
    return {
        "message": "Index rebuild started (no crawling)",
        "note": "Existing scraped_pages.json will be re-processed with text cleaning"
    }





@app.delete("/api/admin/clear")
async def clear_all_data():
    """Clear all scraped data and index."""
    global knowledge_index, rag_engine
    
    admin = get_admin_manager()
    
    # Check if indexing is running
    if admin.current_status in [IndexStatus.SCRAPING, IndexStatus.PROCESSING_PDFS, IndexStatus.INDEXING]:
        raise HTTPException(
            status_code=409,
            detail="Cannot clear while indexing is in progress"
        )
    
    result = admin.clear_all_data()
    
    # Reset global instances
    knowledge_index = None
    rag_engine = RAGEngine()
    
    return {
        "success": True,
        "message": "All data cleared",
        "details": result
    }


# ============== Document Endpoints ==============

@app.get("/api/documents/types")
async def get_document_types():
    """Get available document types for filtering."""
    return {
        "types": [
            {"id": "syllabus", "name": "Syllabus", "description": "Course content and curriculum"},
            {"id": "calendar", "name": "Academic Calendar", "description": "Semester dates and holidays"},
            {"id": "notice", "name": "Notices", "description": "Announcements and circulars"},
            {"id": "rules", "name": "Rules & Regulations", "description": "Policies and guidelines"},
            {"id": "admission", "name": "Admission", "description": "Admission information"},
            {"id": "examination", "name": "Examination", "description": "Exam schedules and patterns"},
            {"id": "fee", "name": "Fee Structure", "description": "Tuition and payment info"},
            {"id": "placement", "name": "Placement", "description": "Career and placement info"},
            {"id": "general", "name": "General", "description": "Other documents"},
            {"id": "webpage", "name": "Webpages", "description": "Website content"}
        ]
    }


# ============== Run Server ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        reload_dirs=["./modules"],       # Only watch source code
        reload_excludes=[
            "venv",                      # 38k+ files — biggest culprit
            "storage",                   # ChromaDB index files and metadata
            "__pycache__",
            "*.pkl",
            "*.faiss",
            "data",
            "cleaned_pages",
        ],
    )
