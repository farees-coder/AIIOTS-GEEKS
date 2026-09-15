from typing import Dict, Callable, Optional
from .crawler import DocumentationCrawler
from .cleaner import clean_html
import asyncio

class Scraper:
    def __init__(
        self, 
        start_url: str, 
        max_pages: int = 100, 
        max_depth: int = 5,
        timeout: int = 30000
    ):
        self.crawler = DocumentationCrawler(
            start_url=start_url,
            max_pages=max_pages,
            max_depth=max_depth,
            timeout=timeout
        )
        
    def set_callbacks(self, on_page_crawled: Optional[Callable] = None, on_error: Optional[Callable] = None):
        self.crawler.on_page_crawled = on_page_crawled
        self.crawler.on_error = on_error

    async def scrape_and_clean(self) -> Dict[str, str]:
        """
        Crawls the documentation and returns cleaned content.
        Returns Dict[url, cleaned_content]
        """
        raw_results = await self.crawler.crawl()
        
        cleaned_results = {}
        for url, html in raw_results.items():
            cleaned_content = clean_html(html)
            if cleaned_content:
                cleaned_results[url] = cleaned_content
                
        return cleaned_results
