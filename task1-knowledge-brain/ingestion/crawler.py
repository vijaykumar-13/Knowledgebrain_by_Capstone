
import asyncio
import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env")

SITEMAP_URL = (
    "https://support.brilliantdirectories.com/support/sitemap.xml"
)

# Resolve relative DATABASE_PATH values against the
# task1-knowledge-brain directory, just like app/main.py.
db_setting = Path(os.getenv("DATABASE_PATH", "knowledge.db"))
DB_PATH = (
    db_setting
    if db_setting.is_absolute()
    else BASE_DIR / db_setting
)

USER_AGENT = "AI-Tech-Consultant-Student-Project/0.1"
REQUEST_DELAY = 1.0
MAX_ARTICLES = 20

RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)


# --------------------------------------------------
# Database
# --------------------------------------------------

def initialize_database():
    """Create the article and ingestion-tracking tables."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                url TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                last_modified TEXT,
                ingested_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active'
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_log (
                url TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                last_attempt TEXT,
                error_message TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )

    print(f"Database path: {DB_PATH}")
    print(f"Database exists: {DB_PATH.exists()}")

    with sqlite3.connect(DB_PATH) as conn:
        tables = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """
        ).fetchall()

        print("Database tables:", [row[0] for row in tables])


def record_ingestion(url, status, error_message=None):
    """Save the URL status, attempt count, and latest error."""
    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO ingestion_log (
                url,
                status,
                attempts,
                last_attempt,
                error_message,
                updated_at
            )
            VALUES (
                ?, ?,
                CASE WHEN ? = 'processing' THEN 1 ELSE 0 END,
                CASE WHEN ? = 'processing' THEN ? ELSE NULL END,
                ?, ?
            )
            ON CONFLICT(url) DO UPDATE SET
                status = excluded.status,
                attempts = ingestion_log.attempts
                    + CASE
                        WHEN excluded.status = 'processing' THEN 1
                        ELSE 0
                      END,
                last_attempt = CASE
                    WHEN excluded.status = 'processing'
                    THEN excluded.last_attempt
                    ELSE ingestion_log.last_attempt
                END,
                error_message = excluded.error_message,
                updated_at = excluded.updated_at
            """,
            (
                url,
                status,
                status,
                status,
                now,
                error_message,
                now,
            ),
        )


def get_ingestion_status():
    """Return the number of URLs recorded for each status."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT status, COUNT(*)
            FROM ingestion_log
            GROUP BY status
            """
        ).fetchall()

    return {status: count for status, count in rows}


def prioritize_urls(all_urls):
    """Prioritize unfinished URLs before previously processed URLs."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT url, status FROM ingestion_log"
        ).fetchall()

    status_by_url = {url: status for url, status in rows}

    unfinished = []
    previously_processed = []

    for url in dict.fromkeys(all_urls):
        status = status_by_url.get(url)

        if status is None or status in {
            "pending",
            "processing",
            "failed",
        }:
            unfinished.append(url)
        else:
            previously_processed.append(url)

    return unfinished + previously_processed


# --------------------------------------------------
# HTTP and sitemap
# --------------------------------------------------

async def fetch(client, url, retries=3):
    """Fetch a URL and retry temporary HTTP or network errors."""
    for attempt in range(retries):
        try:
            response = await client.get(
                url,
                follow_redirects=True,
            )

            if response.status_code in RETRYABLE_STATUS_CODES:
                if attempt == retries - 1:
                    response.raise_for_status()

                await asyncio.sleep(2 ** attempt)
                continue

            response.raise_for_status()
            return response

        except (
            httpx.TimeoutException,
            httpx.NetworkError,
        ):
            if attempt == retries - 1:
                raise

            await asyncio.sleep(2 ** attempt)

    raise RuntimeError(
        f"Could not fetch URL after {retries} attempts: {url}"
    )


async def get_urls(client, sitemap_url, visited=None):
    """Read article URLs from a sitemap and child sitemaps."""
    if visited is None:
        visited = set()

    if sitemap_url in visited:
        return []

    visited.add(sitemap_url)

    response = await fetch(client, sitemap_url)
    root = ElementTree.fromstring(response.content)

    namespace = {
        "sm": "http://www.sitemaps.org/schemas/sitemap/0.9"
    }

    child_sitemaps = root.findall(
        "sm:sitemap/sm:loc",
        namespace,
    )

    if child_sitemaps:
        result = []

        for node in child_sitemaps:
            if not node.text:
                continue

            await asyncio.sleep(REQUEST_DELAY)

            result.extend(
                await get_urls(
                    client,
                    node.text.strip(),
                    visited,
                )
            )

        return result

    return [
        node.text.strip()
        for node in root.findall("sm:url/sm:loc", namespace)
        if node.text and node.text.strip()
    ]


# --------------------------------------------------
# Article parsing
# --------------------------------------------------

def parse_article(html):
    """Extract an article title and readable text from HTML."""
    soup = BeautifulSoup(html, "lxml")

    for element in soup.select(
        "script, style, nav, footer, header, aside, noscript"
    ):
        element.decompose()

    main = (
        soup.select_one("article")
        or soup.select_one("main")
        or soup.select_one(
            ".article-content, .solution-content, "
            ".kb-article, .content-wrapper"
        )
        or soup.body
        or soup
    )

    title_selectors = [
        "article h1",
        "main h1",
        ".article-title",
        ".solution-title",
        ".kb-title",
        "h1",
    ]

    title = None

    for selector in title_selectors:
        heading = main.select_one(selector)

        if heading:
            candidate = heading.get_text(" ", strip=True)

            if (
                candidate
                and candidate.lower()
                != "what do you need help with?"
            ):
                title = candidate
                break

    if not title:
        title = (
            soup.title.get_text(" ", strip=True)
            if soup.title
            else "Untitled"
        )

    content = main.get_text("\n", strip=True)

    return title, content


# --------------------------------------------------
# Save articles
# --------------------------------------------------

def save_article(url, title, content, last_modified=None):
    """Insert an article, update changed content, or skip unchanged."""
    digest = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    now = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as conn:
        old = conn.execute(
            "SELECT content_hash FROM articles WHERE url = ?",
            (url,),
        ).fetchone()

        if old and old[0] == digest:
            return "unchanged"

        conn.execute(
            """
            INSERT INTO articles (
                url,
                title,
                content,
                content_hash,
                last_modified,
                ingested_at,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, 'active')
            ON CONFLICT(url) DO UPDATE SET
                title = excluded.title,
                content = excluded.content,
                content_hash = excluded.content_hash,
                last_modified = excluded.last_modified,
                ingested_at = excluded.ingested_at,
                status = 'active'
            """,
            (
                url,
                title,
                content,
                digest,
                last_modified,
                now,
            ),
        )

    return "updated" if old else "added"


# --------------------------------------------------
# Main crawler
# --------------------------------------------------

async def crawl():
    """Crawl up to MAX_ARTICLES support articles."""
    initialize_database()

    counts = {
        "added": 0,
        "updated": 0,
        "unchanged": 0,
        "failed": 0,
    }

    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    ) as client:
        all_urls = await get_urls(client, SITEMAP_URL)

        # Remove duplicates while preserving order.
        unique_urls = list(dict.fromkeys(all_urls))

        # Keep only expected support article URLs.
        valid_urls = []

        for url in unique_urls:
            parsed = urlparse(url)

            if (
                parsed.scheme == "https"
                and parsed.netloc.lower()
                == "support.brilliantdirectories.com"
                and parsed.path.startswith(
                    "/support/solutions/articles/"
                )
            ):
                valid_urls.append(url)

        prioritized_urls = prioritize_urls(valid_urls)
        urls = prioritized_urls[:MAX_ARTICLES]

        print(f"Valid sitemap article URLs: {len(valid_urls)}")
        print(f"Article processing limit: {MAX_ARTICLES}")
        print(f"URLs selected for this run: {len(urls)}")

        for url in urls:
            try:
                record_ingestion(url, "processing")

                response = await fetch(client, url)
                title, content = parse_article(response.text)

                if not content.strip():
                    raise ValueError("No article text extracted")

                last_modified = response.headers.get("Last-Modified")

                result = save_article(
                    url,
                    title,
                    content,
                    last_modified=last_modified,
                )

                counts[result] += 1
                record_ingestion(url, result)

                print(f"Processed: {title} ({result})")

            except Exception as exc:
                counts["failed"] += 1

                record_ingestion(
                    url,
                    "failed",
                    str(exc),
                )

                print(f"Failed: {url} - {exc}")

            await asyncio.sleep(REQUEST_DELAY)

    with sqlite3.connect(DB_PATH) as conn:
        article_count = conn.execute(
            "SELECT COUNT(*) FROM articles"
        ).fetchone()[0]

    print("Ingestion summary:", counts)
    print("Total articles in database:", article_count)
    print("Saved ingestion status:", get_ingestion_status())


if __name__ == "__main__":
    asyncio.run(crawl())
