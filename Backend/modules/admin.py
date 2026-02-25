"""
CollegeWeb AI - Admin Module
Handles website re-indexing, status monitoring, and admin operations.
"""
import json
import asyncio
from pathlib import Path
from typing import Dict, Optional, Callable
from datetime import datetime
from enum import Enum

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, VECTOR_STORE_DIR, PDF_DIR


class IndexStatus(str, Enum):
    """Status of the indexing process."""
    IDLE = "idle"
    SCRAPING = "scraping"
    PROCESSING_PDFS = "processing_pdfs"
    INDEXING = "indexing"
    COMPLETE = "complete"
    ERROR = "error"


class AdminManager:
    """
    Manages admin operations like re-indexing and status monitoring.
    """
    
    def __init__(self):
        self.current_status = IndexStatus.IDLE
        self.progress = {
            'stage': 'idle',
            'current': 0,
            'total': 0,
            'message': '',
            'started_at': None,
            'completed_at': None,
            'error': None
        }
        self._status_file = DATA_DIR / 'index_status.json'
        self._load_status()
    
    def _load_status(self):
        """Load status from file."""
        if self._status_file.exists():
            try:
                with open(self._status_file, 'r') as f:
                    saved = json.load(f)
                    self.progress.update(saved)
                    # Restore current_status enum from saved stage
                    stage = saved.get('stage', 'idle')
                    if stage == 'complete':
                        self.current_status = IndexStatus.COMPLETE
                    elif stage == 'error':
                        self.current_status = IndexStatus.ERROR
                    elif stage in ['scraping', 'processing_pdfs', 'indexing']:
                        # If finding active status on startup, it means previous run was interrupted
                        print(f"[INFO] Found stale status '{stage}'. Previous run was likely interrupted. Resetting.")
                        self.current_status = IndexStatus.IDLE
                        self.progress['stage'] = 'idle'
                        self.progress['message'] = 'Previous run interrupted. System reset.'
                        self.progress['error'] = 'Process restart detected during indexing'
                        self._save_status()
                    else:
                        self.current_status = IndexStatus.IDLE
                    # Status loaded silently - only log errors or state changes
            except Exception as e:
                print(f"[!] Failed to load status: {e}")
    
    def _save_status(self):
        """Save status to file atomically."""
        try:
            # Include current_status in saved data
            save_data = {
                **self.progress,
                'current_status': self.current_status.value
            }
            
            # Atomic write: write to temp file then rename
            temp_file = self._status_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2)
            
            temp_file.replace(self._status_file)
            
        except Exception as e:
            # Silence simple IO errors to avoid log spam, only print if critical
            pass
    
    def update_progress(self, stage: str, current: int = 0, total: int = 0, message: str = ''):
        """Update progress status."""
        self.progress['stage'] = stage
        self.progress['current'] = current
        self.progress['total'] = total
        self.progress['message'] = message
        self._save_status()
    
    def get_status(self) -> Dict:
        """Get current indexing status."""
        # Re-read from disk to ensure fresh data
        self._load_status()
        return {
            'status': self.current_status.value,
            **self.progress
        }
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the current index."""
        stats = {
            'has_index': False,
            'pages_scraped': 0,
            'pdfs_processed': 0,
            'total_chunks': 0,
            'last_updated': None,
            'target_url': None
        }
        
        # Check scraped pages
        pages_file = DATA_DIR / 'scraped_pages.json'
        if pages_file.exists():
            try:
                with open(pages_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    stats['pages_scraped'] = data.get('pages_count', 0)
                    stats['target_url'] = data.get('base_url')
                    stats['last_updated'] = data.get('scraped_at')
            except:
                pass
        
        # Check processed PDFs
        pdfs_file = DATA_DIR / 'processed_pdfs.json'
        if pdfs_file.exists():
            try:
                with open(pdfs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    stats['pdfs_processed'] = data.get('total_pdfs', 0)
                    stats['total_chunks'] = data.get('total_chunks', 0)
            except:
                pass
        
        # Check if FAISS index exists
        index_file = VECTOR_STORE_DIR / 'faiss_index.bin'
        stats['has_index'] = index_file.exists()
        
        return stats
    
    def get_pdf_list(self) -> list:
        """Get list of indexed PDFs."""
        pdfs = []
        pdfs_file = DATA_DIR / 'processed_pdfs.json'
        
        if pdfs_file.exists():
            try:
                with open(pdfs_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for pdf in data.get('pdfs', []):
                        pdfs.append({
                            'name': pdf['name'],
                            'title': pdf['title'],
                            'type': pdf['doc_type'],
                            'pages': pdf['total_pages'],
                            'chunks': pdf['total_chunks']
                        })
            except:
                pass
        
        return pdfs
    


    
    def clear_all_data(self) -> Dict:
        """Clear all scraped data and index."""
        cleared = {
            'pages_file': False,
            'pdfs_file': False,
            'pdfs_folder': False,
            'index': False
        }
        
        try:
            # Clear scraped pages
            pages_file = DATA_DIR / 'scraped_pages.json'
            if pages_file.exists():
                pages_file.unlink()
                cleared['pages_file'] = True
            
            # Clear processed PDFs file
            pdfs_file = DATA_DIR / 'processed_pdfs.json'
            if pdfs_file.exists():
                pdfs_file.unlink()
                cleared['pdfs_file'] = True
            
            # Clear PDF folder
            for pdf in PDF_DIR.glob('*.pdf'):
                pdf.unlink()
            cleared['pdfs_folder'] = True
            
            # Clear index
            index_file = VECTOR_STORE_DIR / 'faiss_index.bin'
            items_file = VECTOR_STORE_DIR / 'knowledge_items.pkl'
            
            if index_file.exists():
                index_file.unlink()
            if items_file.exists():
                items_file.unlink()
            cleared['index'] = True
            
            # Reset status
            self.current_status = IndexStatus.IDLE
            self.progress = {
                'stage': 'idle',
                'current': 0,
                'total': 0,
                'message': 'Data cleared',
                'started_at': None,
                'completed_at': None,
                'error': None
            }
            self._save_status()
            
        except Exception as e:
            print(f"Error clearing data: {e}")
        
        return cleared


# Singleton instance
_admin_manager: Optional[AdminManager] = None


def get_admin_manager() -> AdminManager:
    """Get or create the admin manager singleton."""
    global _admin_manager
    if _admin_manager is None:
        _admin_manager = AdminManager()
    return _admin_manager


if __name__ == "__main__":
    # Test admin functions
    admin = AdminManager()
    
    print("Current Status:")
    print(json.dumps(admin.get_status(), indent=2))
    
    print("\nIndex Stats:")
    print(json.dumps(admin.get_index_stats(), indent=2))
    
    print("\nPDF List:")
    for pdf in admin.get_pdf_list():
        print(f"  - {pdf['name']} ({pdf['type']})")
