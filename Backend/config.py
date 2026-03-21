"""
CollegeWeb AI - Configuration Settings
"""
import os
import sys
from pathlib import Path

# Force UTF-8 for Windows console to prevent UnicodeEncodeError (e.g. for ₹ symbol)
if sys.stdout and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
PDF_DIR = DATA_DIR / "pdfs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)

# Scraping configuration
SCRAPE_CONFIG = {
    "max_pages": 500,           # Maximum pages to scrape
    "max_depth": 5,             # Maximum crawl depth
    "timeout": 30,              # Request timeout in seconds
    "delay": 0.5,               # Delay between requests
    "user_agent": "CollegeWebAI Bot/1.0",
    "allowed_extensions": [".html", ".htm", ".php", ".asp", ".aspx", ""],
    "pdf_extensions": [".pdf"],
}

# PDF Processing configuration
PDF_CONFIG = {
    "max_file_size_mb": 50,     # Maximum PDF file size
    "chunk_size": 500,          # Characters per chunk
    "chunk_overlap": 100,       # Overlap between chunks
}

# Embedding configuration
EMBEDDING_CONFIG = {
    "model_name": "BAAI/bge-base-en-v1.5",  # Upgraded from bge-small for better retrieval
    "dimension": 768,                         # BGE-base uses 768 dimensions
    "batch_size": 32,                         # Batch size for encoding
}

# CSE Department Detection — used at INDEXING TIME to tag chunks with dept='CSE' or 'OTHER'
# Sub-department keywords are checked FIRST to prevent CSE-DS/CYS from being tagged as plain 'CSE'
CSE_SUB_DEPT_KEYWORDS = {
    'CSE-DS': ['cse-ds', 'cse ds', 'data science', 'ds department'],
    'CSE-CYS': ['cys', 'cyber security', 'cybersecurity', 'cse-cys'],
    'CSE-AIML': ['aiml', 'ai & ml', 'ai and ml', 'cse-aiml', 'artificial intelligence and machine learning'],
    'CSE-IOT': ['iot', 'internet of things', 'cse-iot'],
}

CSE_DEPT_KEYWORDS = [
    'cse', 'computer science', 'c.s.e', 'department of computer science',
    'b.tech cse', 'cse syllabus', 'cse lab', 'cse faculty',
]

# CSE Query Detection — used at RUNTIME to detect CSE-specific questions
CSE_QUERY_KEYWORDS = [
    'cse', 'computer science', 'c.s.e', 'data structures', 'algorithms',
    'software engineering', 'dbms', 'operating systems', 'os',
    'computer networks', 'compiler design', 'machine learning',
    'cse syllabus', 'cse faculty', 'c d naidu', 'manmath nath das',
]

# General Query Keywords — queries that should NOT have CSE boost by default
GENERAL_KEYWORDS = [
    'hostel', 'fee', 'admission', 'principal', 'chairman', 'placement',
    'contact', 'address', 'transport', 'governance', 'leadership', 'director',
]

# Other Branch Keywords — queries that should definitely NOT have CSE boost
OTHER_BRANCH_KEYWORDS = [
    'ece', 'mechanical', 'civil', 'eee', 'it', 'electronics', 'electrical',
]

# RAG configuration
RAG_CONFIG = {
    "top_k": 3,                  # Tight default — DYNAMIC_K in rag_engine.py overrides per query type
    "min_similarity": 0.6,      # Strict threshold to filter PDF noise (313 PDFs vs 103 pages)
    "max_context_length": 15000, # Increased context limit for Gemini/Large Tables
}

# LLM configuration — 4-tier fallback chain (with sub-model fallbacks)
# Priority: Gemini → OpenRouter (mistral-7b → solar-pro-3) → HF (Llama-3-8B → Mistral-v0.3) → Ollama
LLM_CONFIG = {
    # Tier 1 — Google Gemini (fastest, best quality)
    "gemini_model": "gemini-2.0-flash",

    # Tier 2 — OpenRouter (primary + fallback)
    "openrouter_model": "mistralai/mistral-7b-instruct",
    "openrouter_fallback": "upstage/solar-pro-3:free",

    # Tier 3 — HuggingFace (primary + fallback)
    "hf_model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "hf_fallback": "mistralai/Mistral-7B-Instruct-v0.3",

    # Tier 4 — Ollama local (offline safety net, no API key)
    "ollama_model": "qwen2.5:1.5b",

    "temperature": 0.1,   # Low = factual, minimal hallucination
    "max_tokens": 1024,   # Enough for detailed college answers
    "enabled": True,
}

# Firecrawl configuration
FIRECRAWL_CONFIG = {
    "max_pages": 100,              # Maximum pages to crawl
    "include_formats": ["markdown", "html"],
    "exclude_tags": ["nav", "footer", "header", "aside", "script", "style"],
}

# Default target website (can be overridden via API)
DEFAULT_TARGET_URL = "https://vnrvjiet.ac.in/"  # Replace with actual college URL
