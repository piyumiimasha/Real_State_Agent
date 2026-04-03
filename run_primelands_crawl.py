
import sys
import time
from pathlib import Path

# Ensure src is in sys.path
project_root = Path(__file__).parent
src_path = str(project_root / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from context_engineering.application.ingest_document_service.web_crawler import PrimelandsWebCrawler
from context_engineering.config import CRAWL_OUT_DIR

# --- Crawl Configuration ---
BASE_URL = "https://www.primelands.lk/"
START_PATHS = [
    "/", "/lands", "/apartments", "/houses", "/portfolio-properties", "/about-us", "/contact-us", "/news-events", "/faq",
    "/projects", "/offers", "/careers", "/testimonials", "/blog"
]
START_URLS = [BASE_URL.rstrip("/") + path for path in START_PATHS]
EXCLUDE_PATTERNS = [
    "/login", "/terms", "/privacy", "/admin", "/cart", "/user", "/account",
    "/images/", "/downloads/", "/media/", ".jpg", ".png", ".pdf", ".jpeg", ".svg",
    "/sin", "/tam"
]
MAX_DEPTH = 3
REQUEST_DELAY = 2.0
JSONL_PATH = CRAWL_OUT_DIR / "primelands_docs.jsonl"

# --- Run Crawl ---
crawler = PrimelandsWebCrawler(
    base_url=BASE_URL,
    max_depth=MAX_DEPTH,
    exclude_patterns=EXCLUDE_PATTERNS
)

start_time = time.time()
print(f"\n🚀 Starting crawl at {time.strftime('%H:%M:%S')}\n")
documents = crawler.crawl(START_URLS, request_delay=REQUEST_DELAY)

elapsed = time.time() - start_time
print(f"\n✅ Crawl complete in {elapsed:.1f}s")
print(f"📄 Documents collected: {len(documents)}")
print(f"🔗 URLs visited: {len(crawler.visited)}")

# Optionally, save results
def save_jsonl(docs, path):
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


# --- Save markdown files ---
from urllib.parse import urlparse
import json
from context_engineering.config import MARKDOWN_DIR

MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
for i, doc in enumerate(documents):
    url_path = urlparse(doc['url']).path.strip('/').replace('/', '_')
    if not url_path:
        url_path = "homepage"
    filename = f"{i:03d}_{url_path}.md"
    md_file = MARKDOWN_DIR / filename
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(f"# {doc['title']}\n\n")
        f.write(f"**URL**: {doc['url']}\n\n")
        f.write(f"**Depth**: {doc['depth_level']}\n\n")
        f.write("---\n\n")
        f.write(doc['content'])
print(f"✅ Saved {len(documents)} markdown files to {MARKDOWN_DIR}")

# --- Save JSONL ---
def save_jsonl(docs, path):
    with open(path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
save_jsonl(documents, JSONL_PATH)
print(f"✅ Saved JSONL corpus to {JSONL_PATH}")
