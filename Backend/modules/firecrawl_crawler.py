"""
VNRVJIET AI - Firecrawl-based Website Crawler
Professional recursive web crawling with clean markdown output.

Features:
- Recursive crawling with deep nesting support
- Clean markdown output (nav/footer removed)
- Metadata preservation (title, URL, section)
- Rate limiting handled automatically by Firecrawl
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime

from firecrawl import FirecrawlApp

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DATA_DIR, PDF_DIR


@dataclass
class CrawledPage:
    """Represents a crawled webpage with clean markdown content."""
    url: str
    title: str
    markdown: str
    clean_text: str
    links: List[str]
    pdf_links: List[str]
    scraped_at: str
    content_hash: str
    metadata: Dict


class FirecrawlCrawler:
    """
    Professional website crawler using Firecrawl API.
    
    Advantages over BeautifulSoup scraper:
    - Handles JavaScript-rendered pages
    - Recursive crawling with sitemap support
    - Clean markdown output (no nav/footer junk)
    - Automatic rate limiting
    - Better handling of complex nested structures
    
    PDF Handling:
    - Only whitelisted important PDFs are collected
    - Filters out noise (notices, circulars, results, timetables)
    """
    
    # Whitelist: Only index PDFs containing these keywords
    IMPORTANT_PDF_KEYWORDS = [
        "syllabus",
        "regulation",
        "curriculum",
        "brochure",
        "handbook",
        "academic",
        "placement",
        "admission",
        "fee",
        "hostel",
        "department",
        "about",
        "nba",
        "naac",
        "accreditation",
        "faculty",
        "circular"
    ]
    
    # Blacklist: Skip PDFs containing these keywords (noise)
    SKIP_PDF_KEYWORDS = [
        "circular",
        "notice",
        "result",
        "timetable",
        "attendance",
        "newsletter",
        "password",
        "memo",
        "minutes",
        "tender",
        "quotation",
        "scholarship",
        "knowledge",
        "Mtech"
        
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('FIRECRAWL_API_KEY')
        if not self.api_key:
            raise ValueError("Firecrawl API key required. Set FIRECRAWL_API_KEY env variable.")
        
        self.app = FirecrawlApp(api_key=self.api_key)
        self.crawled_pages: List[CrawledPage] = []
        self.pdf_urls: List[str] = []  # Only important PDFs
        self.skipped_pdfs: int = 0  # Count of skipped PDFs
    
    def _is_important_pdf(self, url: str) -> bool:
        """Check if PDF is important based on whitelist/blacklist."""
        url_lower = url.lower()
        
        # First check blacklist - skip if matches
        for skip_word in self.SKIP_PDF_KEYWORDS:
            if skip_word in url_lower:
                return False
        
        # Then check whitelist - include if matches
        for important_word in self.IMPORTANT_PDF_KEYWORDS:
            if important_word in url_lower:
                return True
        
        # Default: skip unknown PDFs (conservative approach)
        return False
    
    def _get_content_hash(self, content: str) -> str:
        """Generate hash of content for deduplication."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _extract_pdf_links(self, markdown: str, base_url: str) -> List[str]:
        """Extract only IMPORTANT PDF links from markdown content."""
        import re
        pdf_pattern = r'\[([^\]]*)\]\(([^)]+\.pdf[^)]*)\)'
        matches = re.findall(pdf_pattern, markdown, re.IGNORECASE)
        pdf_links: List[str] = []
        for _, url in matches:
            # Build full URL
            if url.startswith('http'):
                full_url = url
            elif url.startswith('/'):
                from urllib.parse import urljoin
                full_url = urljoin(base_url, url)
            else:
                continue
            
            # FILTER: Only keep important PDFs
            if self._is_important_pdf(full_url):
                pdf_links.append(full_url)
            else:
                self.skipped_pdfs = self.skipped_pdfs + 1  # pyre-ignore[58]
        
        return list(set(pdf_links))
    
    def _extract_links(self, markdown: str, base_url: str) -> List[str]:
        """Extract non-PDF links from markdown content."""
        import re
        link_pattern = r'\[([^\]]*)\]\(([^)]+)\)'
        matches = re.findall(link_pattern, markdown)
        links: List[str] = []
        for _, url in matches:
            # Skip PDFs and external links
            if '.pdf' in url.lower():
                continue
            if url.startswith('mailto:') or url.startswith('tel:'):
                continue
            if url.startswith('#'):
                continue
                
            if url.startswith('http'):
                links.append(url)
            elif url.startswith('/'):
                from urllib.parse import urljoin
                links.append(urljoin(base_url, url))
        
        return list(set(links))
    
    def crawl_website(self, url: str, max_pages: int = 100, 
                      progress_callback: Optional[Callable] = None) -> Dict:
        """
        Recursively crawl entire website using Firecrawl.
        Uses async crawl with status polling for real-time progress updates.
        
        Args:
            url: Base URL to crawl
            max_pages: Maximum number of pages to crawl
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with crawl statistics
        """
        import time
        
        print(f"[FIRECRAWL] Starting crawl of {url} (max {max_pages} pages)")
        
        # Send initial progress
        if progress_callback:
            progress_callback({
                "stage": "scraping",
                "current": 0,
                "total": max_pages,
                "percent": 0,
                "current_url": f"Starting Firecrawl crawl of {url}..."
            })
        
        try:
            # Use Firecrawl's async crawl API (start_crawl) for real-time progress
            # Start the crawl job
            crawl_response = self.app.start_crawl(
                url,
                limit=max_pages,
                scrape_options={
                    'formats': ['markdown', 'html'],
                }
            )
            
            # Get the crawl ID
            crawl_id = crawl_response.get('id') if isinstance(crawl_response, dict) else getattr(crawl_response, 'id', None)
            
            if not crawl_id:
                print("[FIRECRAWL] Warning: No crawl ID returned, falling back to sync crawl")
                return self._sync_crawl(url, max_pages, progress_callback)
            
            print(f"[FIRECRAWL] Crawl job started with ID: {crawl_id}")
            
            # Poll for status until complete
            pages_crawled = 0
            status = "scraping"
            
            while status in ["scraping", "processing", "pending", "running", "crawling"]:
                time.sleep(2)  # Poll every 2 seconds
                
                try:
                    status_response = self.app.get_crawl_status(crawl_id)
                    
                    if isinstance(status_response, dict):
                        status = status_response.get('status', 'unknown')
                        pages_crawled = status_response.get('completed', 0) or status_response.get('current', 0) or 0
                        total_to_crawl = status_response.get('total', max_pages)
                    else:
                        status = getattr(status_response, 'status', 'unknown')
                        pages_crawled = getattr(status_response, 'completed', 0) or getattr(status_response, 'current', 0) or 0
                        total_to_crawl = getattr(status_response, 'total', max_pages)
                    
                    # Send progress update
                    if progress_callback:
                        percent = round((pages_crawled / max(total_to_crawl, 1)) * 100, 2)
                        progress_callback({
                            "stage": "scraping",
                            "current": pages_crawled,
                            "total": total_to_crawl,
                            "percent": min(percent, 99),  # Cap at 99% until processing
                            "current_url": f"Crawling... {pages_crawled}/{total_to_crawl} pages"
                        })
                    
                    print(f"[FIRECRAWL] Status: {status}, Pages: {pages_crawled}/{total_to_crawl}")
                    
                except Exception as e:
                    print(f"[FIRECRAWL] Status check error: {e}")
                    time.sleep(2)
            
            # Get final results
            if status == "completed":
                crawl_result = self.app.get_crawl_status(crawl_id)
                pages_data = crawl_result.get('data', []) if isinstance(crawl_result, dict) else getattr(crawl_result, 'data', [])
            else:
                print(f"[FIRECRAWL] Crawl ended with status: {status}")
                pages_data: List = []
            return self._process_crawl_results(pages_data, url, progress_callback)
            
        except AttributeError as e:
            # start_crawl may not exist, fall back to sync
            print(f"[FIRECRAWL] Async crawl not available ({e}), using sync crawl...")
            return self._sync_crawl(url, max_pages, progress_callback)
        except Exception as e:
            print(f"[FIRECRAWL ERROR] Crawl failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _sync_crawl(self, url: str, max_pages: int, 
                    progress_callback: Optional[Callable] = None) -> Dict:
        """Fallback synchronous crawl method using crawl()."""
        print(f"[FIRECRAWL] Using synchronous crawl...")
        
        if progress_callback:
            progress_callback({
                "stage": "scraping",
                "current": 0,
                "total": max_pages,
                "percent": 5,
                "current_url": f"Crawling {url} (this may take a few minutes)..."
            })
        
        # Use the correct method name: crawl (not crawl_url)
        crawl_result = self.app.crawl(
            url,
            limit=max_pages,
            scrape_options={
                'formats': ['markdown', 'html'],
            }
        )
        
        pages_data = crawl_result.get('data', []) if isinstance(crawl_result, dict) else getattr(crawl_result, 'data', [])
        return self._process_crawl_results(pages_data, url, progress_callback)
    
    def _process_crawl_results(self, pages_data: List, base_url: str,
                                progress_callback: Optional[Callable] = None) -> Dict:
        """Process crawl results and extract page data."""
        print(f"[FIRECRAWL] Processing {len(pages_data)} crawled pages...")
        
        if not pages_data:
            print("[FIRECRAWL] Warning: No pages returned from crawl")
            if progress_callback:
                progress_callback({
                    "stage": "scraping",
                    "current": 0,
                    "total": 0,
                    "percent": 100,
                    "current_url": "No pages found"
                })
            return {"pages_scraped": 0, "pdfs_found": 0}
        
        total_pages = len(pages_data)
        
        for idx, page_data in enumerate(pages_data):
            try:
                # Handle both dict and Document object types
                if isinstance(page_data, dict):
                    markdown = page_data.get('markdown', '') or ''
                    html = page_data.get('html', '') or ''
                    page_url = page_data.get('url', '') or page_data.get('sourceURL', '')
                    metadata = page_data.get('metadata', {}) or {}
                else:
                    # Document object - use attribute access
                    markdown = getattr(page_data, 'markdown', '') or ''
                    html = getattr(page_data, 'html', '') or ''
                    page_url = getattr(page_data, 'url', '') or getattr(page_data, 'sourceURL', '') or ''
                    metadata = getattr(page_data, 'metadata', {}) or {}
                
                # Get title from metadata
                if isinstance(metadata, dict):
                    title = metadata.get('title', '') or metadata.get('ogTitle', '') or page_url
                else:
                    title = getattr(metadata, 'title', '') or getattr(metadata, 'ogTitle', '') or page_url
                
                # Skip pages with no content
                if len(markdown) < 50:
                    continue
                
                # Extract clean text from markdown
                clean_text = self._markdown_to_text(markdown)
                
                # Extract links and PDFs
                pdf_links = self._extract_pdf_links(markdown, base_url)
                page_links = self._extract_links(markdown, base_url)
                
                # Collect PDF URLs
                self.pdf_urls.extend(pdf_links)
                
                # Get metadata values safely
                if isinstance(metadata, dict):
                    desc = metadata.get('description', '')
                    keywords = metadata.get('keywords', '')
                    language = metadata.get('language', 'en')
                else:
                    desc = getattr(metadata, 'description', '') or ''
                    keywords = getattr(metadata, 'keywords', '') or ''
                    language = getattr(metadata, 'language', 'en') or 'en'
                
                # Create crawled page object
                crawled_page = CrawledPage(
                    url=page_url,
                    title=title,
                    markdown=markdown,
                    clean_text=clean_text,
                    links=page_links,
                    pdf_links=pdf_links,
                    scraped_at=datetime.now().isoformat(),
                    content_hash=self._get_content_hash(clean_text),
                    metadata={
                        'description': desc,
                        'keywords': keywords,
                        'language': language,
                    }
                )
                
                self.crawled_pages.append(crawled_page)
                
                # Progress callback during processing phase
                if progress_callback:
                    progress_callback({
                        "stage": "scraping",
                        "current": idx + 1,
                        "total": total_pages,
                        "percent": round(((idx + 1) / total_pages) * 100, 2),
                        "current_url": f"Processing: {title[:40]}..." if len(title) > 40 else f"Processing: {title}"
                    })
            except Exception as e:
                print(f"[FIRECRAWL] Error processing page {idx}: {e}")
                continue
        
        # Deduplicate PDF URLs
        self.pdf_urls = list(set(self.pdf_urls))
        
        print(f"[FIRECRAWL] Processed {len(self.crawled_pages)} pages, found {len(self.pdf_urls)} PDFs")
        
        return {
            "pages_scraped": len(self.crawled_pages),
            "pdfs_found": len(self.pdf_urls)
        }
    
    def _markdown_to_text(self, markdown: str) -> str:
        """Convert markdown to clean plain text."""
        import re
        
        text = markdown
        
        # Remove markdown links but keep text: [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove markdown images
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
        
        # Remove headers syntax but keep text: ## Header -> Header
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
        
        # Remove bold/italic markers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Remove code blocks
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove horizontal rules
        text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
        
        # Remove bullet points but keep content
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        
        # Clean up excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def download_pdfs(self, progress_callback: Optional[Callable] = None) -> Dict:
        """Download all discovered PDF files with retry logic and exponential backoff."""
        import requests
        import time
        
        total = len(self.pdf_urls)
        downloaded: int = 0
        failed: int = 0
        
        print(f"[FIRECRAWL] Downloading {total} PDFs...")
        
        for idx, pdf_url in enumerate(self.pdf_urls):
            # Extract filename from URL
            filename = pdf_url.split('/')[-1].split('?')[0]
            if not filename.endswith('.pdf'):
                filename += '.pdf'
            
            filepath = PDF_DIR / filename
            
            if filepath.exists():
                print(f"[SKIP] {filename} already exists")
                downloaded = downloaded + 1  # pyre-ignore[58]  # Count existing files
            else:
                # Retry logic: 3 attempts with exponential backoff
                max_retries = 3
                success = False
                
                for attempt in range(1, max_retries + 1):
                    try:
                        timeout = 60 + (attempt * 30)  # Progressive timeout: 90s, 120s, 150s
                        print(f"[ATTEMPT {attempt}/{max_retries}] Downloading {filename} (timeout: {timeout}s)...")
                        
                        response = requests.get(pdf_url, timeout=timeout, stream=True)
                        
                        if response.status_code == 200:
                            # Check file size (max 50MB)
                            content_length = response.headers.get('content-length')
                            if content_length and int(content_length) > 50 * 1024 * 1024:
                                print(f"[SKIP] {filename} exceeds 50MB size limit")
                                break
                            
                            # Download with streaming to handle large files
                            with open(filepath, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                            
                            downloaded = downloaded + 1  # pyre-ignore[58]
                            success = True
                            print(f"[OK] Downloaded {filename}")
                            break
                        else:
                            print(f"[WARN] HTTP {response.status_code} for {filename}")
                            if attempt < max_retries:
                                time.sleep(2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
                    
                    except requests.exceptions.Timeout:
                        print(f"[TIMEOUT] Attempt {attempt} timed out after {timeout}s")
                        if attempt < max_retries:
                            wait_time = 2 ** attempt
                            print(f"[RETRY] Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                        else:
                            print(f"[FAILED] {filename} - Max retries exceeded (connection timeout)")
                            failed = failed + 1  # pyre-ignore[58]
                    
                    except requests.exceptions.ConnectionError as e:
                        print(f"[CONNECTION ERROR] Attempt {attempt}: {str(e)[:100]}")
                        if attempt < max_retries:
                            wait_time = 2 ** attempt
                            print(f"[RETRY] Waiting {wait_time}s before retry...")
                            time.sleep(wait_time)
                        else:
                            print(f"[FAILED] {filename} - Max retries exceeded (connection error)")
                            failed = failed + 1  # pyre-ignore[58]
                    
                    except Exception as e:
                        print(f"[ERROR] Attempt {attempt} failed: {str(e)[:100]}")
                        if attempt < max_retries:
                            time.sleep(2 ** attempt)
                        else:
                            print(f"[FAILED] {filename} - {type(e).__name__}: {str(e)[:100]}")
                            failed = failed + 1  
                        break
            
            # Progress callback
            if progress_callback:
                progress_callback({
                    "stage": "downloading_pdfs",
                    "current": idx + 1,
                    "total": total,
                    "percent": round(((idx + 1) / total) * 100, 2) if total > 0 else 100,
                    "file": filename
                })
        
        print(f"\n[SUMMARY] Total: {total} | Downloaded: {downloaded} | Failed: {failed}")
        return {"total": total, "downloaded": downloaded, "failed": failed}
    
    def save_data(self) -> str:
        """Save crawled data to JSON file."""
        output_file = DATA_DIR / 'scraped_pages.json'
        
        data = {
            'base_url': self.crawled_pages[0].url if self.crawled_pages else '',
            'scraped_at': datetime.now().isoformat(),
            'pages_count': len(self.crawled_pages),
            'pdf_urls': self.pdf_urls,
            'pages': [asdict(page) for page in self.crawled_pages]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Saved {len(self.crawled_pages)} pages to {output_file}")
        return str(output_file)
    
    def load_data(self) -> bool:
        """Load previously crawled data."""
        input_file = DATA_DIR / 'scraped_pages.json'
        
        if not input_file.exists():
            return False
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.pdf_urls = data.get('pdf_urls', [])
            self.crawled_pages = []
            
            for page in data.get('pages', []):
                self.crawled_pages.append(CrawledPage(**page))
            
            print(f"[OK] Loaded {len(self.crawled_pages)} pages from cache")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            return False


# For backward compatibility - can be used as drop-in replacement
def run_full_scrape(target_url: str, max_pages: int = 100, 
                    api_key: Optional[str] = None) -> Dict:
    """
    Run a complete website scrape using Firecrawl.
    Drop-in replacement for the old scraper.run_full_scrape function.
    """
    crawler = FirecrawlCrawler(api_key)
    
    # Crawl website
    crawl_stats = crawler.crawl_website(target_url, max_pages=max_pages)
    
    # Download PDFs
    pdf_stats = crawler.download_pdfs()
    
    # Save data
    data_file = crawler.save_data()
    
    return {
        'crawler': crawler,
        'crawl_stats': crawl_stats,
        'pdf_stats': pdf_stats,
        'data_file': data_file,
        'pages': crawler.crawled_pages,
        'pdf_urls': crawler.pdf_urls
    }


if __name__ == "__main__":
    # Test the crawler
    import sys
    
    api_key = os.getenv('FIRECRAWL_API_KEY') or "fc-3650de2fdc7a4e99823d03cda2ac7a4e"
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://vnrvjiet.ac.in"
    
    result = run_full_scrape(url, max_pages=5, api_key=api_key)
    print(f"\nCrawl Results:")
    print(f"Pages: {result['crawl_stats']['pages_scraped']}")
    print(f"PDFs: {result['pdf_stats']['downloaded']}")
