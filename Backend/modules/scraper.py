"""
CollegeWeb AI - Website Scraping Module
Crawls all website pages, extracts content, and discovers PDF links.
"""
import re
import time
import hashlib
import json
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import Optional, Set, Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SCRAPE_CONFIG, DATA_DIR, PDF_DIR


@dataclass
class ScrapedPage:
    """Represents a scraped webpage."""
    url: str
    title: str
    content: str
    clean_text: str
    links: List[str]
    pdf_links: List[str]
    scraped_at: str
    content_hash: str


class WebsiteScraper:
    """
    Intelligent website scraper that crawls all pages and discovers PDFs.
    """
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.domain = urlparse(base_url).netloc
        self.visited_urls: Set[str] = set()
        self.pdf_urls: Set[str] = set()
        self.scraped_pages: List[ScrapedPage] = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': SCRAPE_CONFIG['user_agent'],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        
    def _get_content_hash(self, content: str) -> str:
        """Generate hash of content for deduplication."""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and within the target domain."""
        try:
            parsed = urlparse(url)
            # Must be same domain
            if parsed.netloc and parsed.netloc != self.domain:
                return False
            # Skip common non-content URLs
            skip_patterns = [
                '/wp-admin', '/admin', '/login', '/logout',
                '/cart', '/checkout', 'javascript:', 'mailto:', 'tel:',
                '#', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.css', '.js',
                '.zip', '.rar', '.exe', '.mp4', '.mp3', '.doc', '.docx', '.xls', '.xlsx'
            ]
            url_lower = url.lower()
            for pattern in skip_patterns:
                if pattern in url_lower:
                    return False
            return True
        except Exception:
            return False
    
    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF file."""
        return url.lower().endswith('.pdf')
    
    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text content from HTML."""
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 
                            'aside', 'form', 'noscript', 'iframe']):
            element.decompose()
        
        # Get main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|main|body', re.I))
        
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)
        
        # Clean up the text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = '\n'.join(lines)
        
        # Remove excessive whitespace
        clean_text = re.sub(r'\n{3,}', '\n\n', clean_text)
        
        return clean_text
    
    def _extract_links(self, soup: BeautifulSoup, current_url: str) -> tuple:
        """Extract all links from the page."""
        page_links = []
        pdf_links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href'].strip()
            
            # Skip empty or fragment-only links
            if not href or href.startswith('#'):
                continue
            
            # Convert to absolute URL
            absolute_url = urljoin(current_url, href)
            
            # Remove fragments
            absolute_url = absolute_url.split('#')[0]
            
            if self._is_pdf_url(absolute_url):
                pdf_links.append(absolute_url)
            elif self._is_valid_url(absolute_url):
                page_links.append(absolute_url)
        
        return list(set(page_links)), list(set(pdf_links))
    
    def scrape_page(self, url: str) -> Optional[ScrapedPage]:
        """Scrape a single page."""
        try:
            response = self.session.get(
                url, 
                timeout=SCRAPE_CONFIG['timeout'],
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                return None
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text(strip=True) if title_tag else url
            
            # Extract content
            clean_text = self._extract_text(soup)
            
            # Skip pages with very little content
            if len(clean_text) < 100:
                return None
            
            # Extract links
            page_links, pdf_links = self._extract_links(soup, url)
            
            # Create scraped page object
            page = ScrapedPage(
                url=url,
                title=title,
                content=response.text,
                clean_text=clean_text,
                links=page_links,
                pdf_links=pdf_links,
                scraped_at=datetime.now().isoformat(),
                content_hash=self._get_content_hash(clean_text)
            )
            
            return page
            
        except requests.RequestException as e:
            print(f"Error scraping {url}: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error scraping {url}: {e}")
            return None
    
    def crawl(self, max_pages=300, progress_callback=None):
        """Crawl website pages starting from base_url."""
        visited = set()
        queue = [self.base_url]
        total = max_pages
        count = 0

        while queue and count < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            page = self.scrape_page(url)
            if page:
                self.scraped_pages.append(page)
                self.pdf_urls.update(page.pdf_links)
                queue.extend([l for l in page.links if l not in visited])
                count += 1

            # Call progress callback INSIDE the loop
            if progress_callback:
                progress_callback({
                    "stage": "scraping",
                    "current": count,
                    "total": total,
                    "percent": round((count / total) * 100, 2),
                    "current_url": url
                })
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.2)

        self.visited_urls = visited
        return {
            "pages_scraped": len(self.scraped_pages),
            "pdfs_found": len(self.pdf_urls)
        }

    def download_pdfs(self, progress_callback=None):
        """Download all discovered PDF files."""
        total = len(self.pdf_urls)
        count = 0
        downloaded = 0

        for pdf_url in self.pdf_urls:
            try:
                name = pdf_url.split("/")[-1]
                path = PDF_DIR / name
                count += 1
                
                if not path.exists():
                    r = self.session.get(pdf_url, timeout=60)
                    if r.status_code == 200 and len(r.content) < 40 * 1024 * 1024:
                        with open(path, "wb") as f:
                            f.write(r.content)
                        downloaded += 1

                if progress_callback:
                    progress_callback({
                        "stage": "downloading_pdfs",
                        "current": count,
                        "total": total,
                        "percent": round((count / total) * 100, 2) if total > 0 else 100,
                        "file": name
                    })
            except Exception as e:
                print(f"Error downloading PDF {pdf_url}: {e}")
        
        return {
            "total": total,
            "downloaded": downloaded
        }

    
    def save_data(self) -> str:
        """Save scraped data to JSON file."""
        output_file = DATA_DIR / 'scraped_pages.json'
        
        data = {
            'base_url': self.base_url,
            'scraped_at': datetime.now().isoformat(),
            'pages_count': len(self.scraped_pages),
            'pdf_urls': list(self.pdf_urls),
            'pages': [asdict(page) for page in self.scraped_pages]
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] Saved scraped data to {output_file}")
        return str(output_file)
    
    def load_data(self) -> bool:
        """Load previously scraped data."""
        input_file = DATA_DIR / 'scraped_pages.json'
        
        if not input_file.exists():
            return False
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.pdf_urls = set(data.get('pdf_urls', []))
            self.scraped_pages = [
                ScrapedPage(**page) for page in data.get('pages', [])
            ]
            
            print(f"[OK] Loaded {len(self.scraped_pages)} pages from cache")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False


def run_full_scrape(target_url: str, max_pages: int = 100) -> Dict:
    """
    Run a complete website scrape operation.
    
    Args:
        target_url: The base URL to scrape
        max_pages: Maximum number of pages to scrape
        
    Returns:
        Dictionary with scraping results and statistics
    """
    scraper = WebsiteScraper(target_url)
    
    # Crawl website
    crawl_stats = scraper.crawl(max_pages=max_pages)
    
    # Download PDFs
    pdf_stats = scraper.download_pdfs()
    
    # Save data
    data_file = scraper.save_data()
    
    return {
        'scraper': scraper,
        'crawl_stats': crawl_stats,
        'pdf_stats': pdf_stats,
        'data_file': data_file,
        'pages': scraper.scraped_pages,
        'pdf_urls': list(scraper.pdf_urls)
    }


if __name__ == "__main__":
    # Test scraping
    import sys
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://example.edu"
    
    results = run_full_scrape(url, max_pages=10)
    print(f"\nScraping Results:")
    print(f"Pages: {results['crawl_stats']['pages_scraped']}")
    print(f"PDFs: {results['pdf_stats']['downloaded']}")
