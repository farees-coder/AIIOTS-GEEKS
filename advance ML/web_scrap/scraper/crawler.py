import os
import asyncio
from typing import List, Set, Dict, Optional, Callable
from playwright.async_api import async_playwright, Page, Browser
from .link_extractor import extract_links

class DocumentationCrawler:
    def __init__(
        self, 
        start_url: str, 
        max_pages: int = 100, 
        max_depth: int = 5,
        timeout: int = 30000
    ):
        self.start_url = start_url
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.timeout = timeout
        
        self.visited: Set[str] = set()
        self.queue: List[Dict] = [{"url": start_url, "depth": 0}]
        
        # Callbacks for progress tracking
        self.on_page_crawled: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

    async def crawl(self) -> Dict[str, str]:
        """
        Crawls the documentation starting from start_url.
        Returns a dictionary of {url: html_content}.
        """
        results: Dict[str, str] = {}
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = await context.new_page()
            
            while self.queue and len(self.visited) < self.max_pages:
                current = self.queue.pop(0)
                url = current["url"]
                depth = current["depth"]
                
                if url in self.visited:
                    continue
                    
                if depth > self.max_depth:
                    continue
                    
                self.visited.add(url)
                
                try:
                    await page.goto(url, timeout=self.timeout, wait_until="domcontentloaded")
                    # Wait a bit for potential JS rendering
                    await page.wait_for_timeout(1000)
                    
                    html = await page.content()
                    results[url] = html
                    
                    if self.on_page_crawled:
                        self.on_page_crawled(url, len(self.visited), self.queue)
                        
                    # Extract and queue new links
                    if depth < self.max_depth:
                        new_links = extract_links(html, url, self.start_url)
                        for link in new_links:
                            if link not in self.visited:
                                # Ensure we don't add duplicates to the queue that are already in queue
                                if not any(item["url"] == link for item in self.queue):
                                    self.queue.append({"url": link, "depth": depth + 1})
                                    
                except Exception as e:
                    if self.on_error:
                        self.on_error(url, str(e))
                        
            await browser.close()
            
        return results
