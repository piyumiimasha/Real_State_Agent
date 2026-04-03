
"""
Primelands Real Estate Web Crawler Service

Provides:
- PrimelandsWebCrawler: Playwright-based async crawler for JS-rendered real estate sites
- Content extraction with BeautifulSoup
- Markdown conversion with markdownify
- Polite crawling with depth control and rate limiting
"""

from typing import List, Dict, Any, Set
from collections import deque
import re
import asyncio
import traceback
from urllib.parse import urljoin, urlparse

from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from bs4.element import Tag
from markdownify import markdownify as md


class PrimelandsWebCrawler:
    """
    Async web crawler for Primelands Real Estate using Playwright for JavaScript-rendered content.

    Features:
    - Respects depth limits and exclude patterns
    - Handles SPA/React apps with proper JS rendering waits
    - Extracts clean markdown content
    - Discovers internal links for BFS traversal
    - Polite crawling with configurable delays

    Usage:
        crawler = PrimelandsWebCrawler(
            base_url="https://www.primelands.lk/",
            max_depth=3,
            exclude_patterns=["/login", "/admin", "/cart", "/user", "/account"]
        )
        documents = crawler.crawl(["https://www.primelands.lk/"])
    """
    
    def __init__(self, base_url: str = "https://www.primelands.lk/", max_depth: int = 3, exclude_patterns: List[str] = None):
        self.base_url = base_url
        self.max_depth = max_depth
        self.exclude_patterns = exclude_patterns if exclude_patterns is not None else ["/login", "/admin", "/cart", "/user", "/account"]
        self.visited: Set[str] = set()
        self.documents: List[Dict[str, Any]] = []
    
    def should_crawl(self, url: str) -> bool:
        """Check if URL should be crawled based on rules."""
        if url in self.visited:
            return False
        
        # Must be within base domain
        if not url.startswith(self.base_url):
            return False
        
        # Check exclude patterns
        for pattern in self.exclude_patterns:
            if pattern in url:
                return False
        
        # Skip media files
        if re.search(r'\.(jpg|jpeg|png|gif|pdf|zip|exe)$', url, re.I):
            return False
        
        return True

    def _extract_links(self, soup: BeautifulSoup, url: str) -> List[str]:
        links: List[str] = []
        for a in soup.find_all('a', href=True):
            if not isinstance(a, Tag):
                continue
            href = a.get('href', '')
            if not href:
                continue
            if href.startswith('/'):
                href = self.base_url + href
            elif not href.startswith('http'):
                href = urljoin(url, href)
            if href.startswith(self.base_url):
                href = href.split('#')[0].split('?')[0]
                if href and href != url:
                    links.append(href)
        return list(set(links))

    def _strip_noise(self, soup: BeautifulSoup) -> None:
        for element in soup(["script", "style", "nav", "footer", "aside", "noscript", "iframe", "header"]):
            element.decompose()

        noise_pattern = re.compile(r"(nav|menu|header|footer|breadcrumb|cookie|popup|modal|sidebar|social|lang|language)", re.I)
        for element in soup.find_all(True):
            if not isinstance(element, Tag):
                continue
            if element.attrs is None:
                continue
            element_id = element.get("id") or ""
            class_list = " ".join(element.get("class") or [])
            if noise_pattern.search(element_id) or noise_pattern.search(class_list):
                element.decompose()

    def _select_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        candidates = soup.select("main, article, section, div")
        if not candidates:
            return soup.body or soup

        def score(el: BeautifulSoup) -> int:
            text = el.get_text(" ", strip=True)
            has_h1 = 1 if el.find("h1") else 0
            return len(text) + (has_h1 * 500)

        best = max(candidates, key=score)
        return best
    
    def extract_content(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """
        Extract clean content from HTML soup, focusing on property detail pages.
        Returns dict with:
        - title: Page title
        - headings: List of h1-h4 text
        - content: Clean markdown
        - links: List of internal URLs
        """
        links = self._extract_links(soup, url)

        # Remove noise elements before content extraction
        self._strip_noise(soup)

        # Only extract/save content for property detail pages (land, apartment, house, portfolio-property) with any language code
        # e.g., https://www.primelands.lk/land/PROPERTY-NAME/en or /sin or /tam
        is_detail_page = re.search(r"/(land|apartment|house|portfolio-property)/[^/]+/(en|sin|tam)$", url)
        if is_detail_page:
            # Prefer the main content container for detail pages
            main_content = self._select_main_content(soup)
            # Get title
            title = soup.title.string if soup.title else url.split("/")[-2]
            title = title.strip() if title else "Untitled"
            # Extract headings
            headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2', 'h3', 'h4'])]
            # Extract markdown
            if main_content:
                content_md = md(str(main_content), heading_style="ATX")
            else:
                content_md = md(str(soup), heading_style="ATX")
            # Clean up markdown
            content_md = re.sub(r'You need to enable JavaScript.*?\\.', '', content_md, flags=re.IGNORECASE)
            content_md = re.sub(r'\\n{3,}', '\\n\\n', content_md)
            content_md = content_md.strip()
            # Save whatever is extracted, regardless of length
            return {
                "title": title,
                "headings": headings,
                "content": content_md,
                "links": list(set(links))
            }
        else:
            # For listing/navigation pages, just return links (no content)
            return {
                "title": "",
                "headings": [],
                "content": "",
                "links": list(set(links))
            }
    
    async def crawl_async(self, start_urls: List[str], request_delay: float = 2.0) -> List[Dict[str, Any]]:
        """
        BFS crawl with Playwright for JS rendering.
        
        Args:
            start_urls: List of seed URLs to start crawling
            request_delay: Seconds to wait between requests (politeness)
        
        Returns:
            List of document dicts with url, title, content, links, depth_level
        """
        queue = deque([(url, 0) for url in start_urls])
        
        async with async_playwright() as p:
            # Launch browser (headless mode)
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            page.set_default_timeout(30000)  # 30 seconds
            page.set_default_navigation_timeout(90000)

            async def block_heavy_assets(route):
                resource_type = route.request.resource_type
                if resource_type in {"image", "media", "font"}:
                    await route.abort()
                    return
                url = route.request.url.lower()
                if re.search(r"\.(jpg|jpeg|png|gif|svg|webp|mp4|avi|mov|woff|woff2|ttf)$", url):
                    await route.abort()
                    return
                await route.continue_()

            await page.route("**/*", block_heavy_assets)
            
            while queue:
                url, depth = queue.popleft()
                
                if depth > self.max_depth or not self.should_crawl(url):
                    continue
                
                try:
                    print(f"🔍 [{depth}] {url}")
                    self.visited.add(url)
                    
                    # Navigate and wait for DOM to be ready (networkidle can hang on heavy sites)
                    await page.goto(url, wait_until="domcontentloaded", timeout=90000)
                    
                    # Wait for property detail content to render
                    try:
                        # Wait for a likely property detail selector (adjust as needed)
                        await page.wait_for_selector(".property-details, .main-content, .property-info, main, article", timeout=15000)
                        await page.wait_for_timeout(2000)  # Additional wait for JS
                        # Scroll to trigger lazy loading
                        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        await page.wait_for_timeout(1000)
                    except:
                        # Fallback: just wait longer
                        await page.wait_for_timeout(7000)
                    
                    # Get rendered HTML
                    html = await page.content()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract content
                    doc_data = self.extract_content(soup, url)
                    doc_data['url'] = url
                    doc_data['depth_level'] = depth
                    
                    # Only save if content is substantial
                    if len(doc_data['content']) >= 100:
                        self.documents.append(doc_data)
                        print(f"   ✅ Saved ({len(doc_data['content'])} chars, {len(doc_data['links'])} links found)")
                    else:
                        print(f"   ⚠️  Skipped (content too short: {len(doc_data['content'])} chars)")
                    
                    # Add links to queue if depth allows
                    if depth < self.max_depth:
                        links_added = 0
                        for link in doc_data['links']:
                            if link not in self.visited and link not in [item[0] for item in queue]:
                                queue.append((link, depth + 1))
                                links_added += 1
                        if links_added > 0:
                            print(f"   📎 Added {links_added} new URLs to queue (depth {depth + 1})")
                    
                    # Progress update
                    print(f"   📊 Progress: {len(self.documents)} docs saved, {len(self.visited)} visited, {len(queue)} in queue")
                    
                    # Polite delay
                    await asyncio.sleep(request_delay)
                    
                except Exception as e:
                    error_msg = str(e)
                    if "404" in error_msg or "net::ERR_" in error_msg:
                        print(f"   ⚠️  Page not found (404) - skipping")
                    else:
                        print(f"   ❌ Error: {error_msg[:100]}")
                        traceback.print_exc()
                    continue
            
            await browser.close()
        
        return self.documents
    
    def crawl(self, start_urls: List[str], request_delay: float = 2.0) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for async crawl (for Jupyter compatibility).
        
        Args:
            start_urls: List of seed URLs
            request_delay: Seconds between requests
        
        Returns:
            List of crawled documents
        """
        return asyncio.run(self.crawl_async(start_urls, request_delay))


__all__ = ['PrimelandsWebCrawler']

