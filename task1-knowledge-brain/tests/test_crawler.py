
import sqlite3

from ingestion import crawler
from ingestion.crawler import parse_article


def test_parse_article_extracts_title_and_content():
    html = """
    <html>
      <head><title>Example Support Article</title></head>
      <body>
        <nav>Navigation that should be removed</nav>
        <main>
          <h1>Import Members</h1>
          <p>Download the CSV template.</p>
          <p>Keep the column headers unchanged.</p>
        </main>
        <footer>Footer that should be removed</footer>
      </body>
    </html>
    """

    title, content = parse_article(html)

    assert title == "Import Members"
    assert "Download the CSV template." in content
    assert "Keep the column headers unchanged." in content
    assert "Navigation that should be removed" not in content
    assert "Footer that should be removed" not in content


def test_parse_article_uses_fallback_title():
    html = """
    <html>
      <head><title>Example Page</title></head>
      <body>
        <main>
          <p>Example support content.</p>
        </main>
      </body>
    </html>
    """

    title, content = parse_article(html)

    assert title == "Example Page"
    assert "Example support content." in content


def test_save_article_adds_then_skips_unchanged(
    tmp_path, monkeypatch
):
    test_db = tmp_path / "test_knowledge.db"
    monkeypatch.setattr(crawler, "DB_PATH", test_db)

    crawler.initialize_database()

    first_result = crawler.save_article(
        "https://example.com/article-1",
        "Test Article",
        "Original article content",
    )

    second_result = crawler.save_article(
        "https://example.com/article-1",
        "Test Article",
        "Original article content",
    )

    assert first_result == "added"
    assert second_result == "unchanged"

    with sqlite3.connect(test_db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM articles"
        ).fetchone()[0]

    assert count == 1


def test_save_article_updates_changed_content(
    tmp_path, monkeypatch
):
    test_db = tmp_path / "test_knowledge.db"
    monkeypatch.setattr(crawler, "DB_PATH", test_db)

    crawler.initialize_database()

    crawler.save_article(
        "https://example.com/article-1",
        "Test Article",
        "Original content",
    )

    result = crawler.save_article(
        "https://example.com/article-1",
        "Updated Article",
        "Updated content",
    )

    assert result == "updated"

    with sqlite3.connect(test_db) as conn:
        row = conn.execute(
            "SELECT title, content FROM articles "
            "WHERE url = ?",
            ("https://example.com/article-1",),
        ).fetchone()

    assert row == ("Updated Article", "Updated content")


def test_prioritize_urls_retries_failed_urls_first(
    tmp_path, monkeypatch
):
    test_db = tmp_path / "test_knowledge.db"
    monkeypatch.setattr(crawler, "DB_PATH", test_db)

    crawler.initialize_database()

    crawler.record_ingestion(
        "https://example.com/old",
        "unchanged",
    )

    crawler.record_ingestion(
        "https://example.com/failed",
        "failed",
        "Temporary network error",
    )

    urls = [
        "https://example.com/old",
        "https://example.com/new",
        "https://example.com/failed",
    ]

    result = crawler.prioritize_urls(urls)

    assert result == [
        "https://example.com/new",
        "https://example.com/failed",
        "https://example.com/old",
    ]


def test_prioritize_urls_deduplicates_and_preserves_new_urls(
    tmp_path, monkeypatch
):
    test_db = tmp_path / "test_knowledge.db"
    monkeypatch.setattr(crawler, "DB_PATH", test_db)

    crawler.initialize_database()

    urls = [
        "https://example.com/article-1",
        "https://example.com/article-2",
        "https://example.com/article-1",
        "https://example.com/article-3",
    ]

    result = crawler.prioritize_urls(urls)

    assert result == [
        "https://example.com/article-1",
        "https://example.com/article-2",
        "https://example.com/article-3",
    ]
