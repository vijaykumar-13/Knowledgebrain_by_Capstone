
import os
import re
import json
import sqlite3
import logging
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from google import genai

try:
    from ingestion.crawler import crawl
except ImportError:
    crawl = None


# --------------------------------------------------
# Configuration
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
DATABASE_PATH = Path(
    os.getenv("DATABASE_PATH", str(BASE_DIR / "knowledge.db"))
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

app = FastAPI(
    title="AI Tech Consultant — Knowledge Brain",
    version="0.5.0",
    description="Knowledge retrieval and question answering API",
)

ingestion_lock = threading.Lock()
ingestion_running = False
ingestion_last_result = "never_run"


# --------------------------------------------------
# Request models
# --------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


class EscalationRequest(BaseModel):
    question: str
    reason: str = "Answer requires human review"


# --------------------------------------------------
# Database helpers
# --------------------------------------------------

def get_connection():
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Database does not exist: {DATABASE_PATH}"
        )

    connection = sqlite3.connect(str(DATABASE_PATH), timeout=30)
    connection.row_factory = sqlite3.Row
    return connection


def get_article_columns(connection):
    rows = connection.execute("PRAGMA table_info(articles)").fetchall()
    return {row["name"].lower(): row["name"] for row in rows}


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{2,}", text.lower()))


# --------------------------------------------------
# Keyword retrieval
# --------------------------------------------------

def search_articles(question: str, top_k: int = 5) -> list[dict]:
    if not DATABASE_PATH.exists():
        logger.error("Database missing: %s", DATABASE_PATH)
        return []

    try:
        with get_connection() as connection:
            columns = get_article_columns(connection)

            if not columns:
                logger.error("Articles table does not exist.")
                return []

            title_col = next(
                (columns[c] for c in ("title", "name") if c in columns),
                None,
            )
            content_col = next(
                (
                    columns[c]
                    for c in (
                        "content", "body", "text",
                        "article_text", "description"
                    )
                    if c in columns
                ),
                None,
            )
            url_col = next(
                (
                    columns[c]
                    for c in ("url", "source_url", "link")
                    if c in columns
                ),
                None,
            )

            if not title_col or not content_col or not url_col:
                logger.error(
                    "Unexpected articles schema: %s",
                    list(columns.keys()),
                )
                return []

            query = (
                f'SELECT "{title_col}" AS title, '
                f'"{content_col}" AS content, '
                f'"{url_col}" AS url '
                f'FROM articles'
            )

            rows = connection.execute(query).fetchall()

        question_tokens = tokenize(question)
        if not question_tokens:
            return []

        matches = []

        for row in rows:
            title = str(row["title"] or "")
            content = str(row["content"] or "")
            url = str(row["url"] or "")

            title_tokens = tokenize(title)
            content_tokens = tokenize(content)

            title_overlap = len(question_tokens & title_tokens)
            content_overlap = len(question_tokens & content_tokens)

            score = (title_overlap * 3) + content_overlap

            # Give a modest boost to exact topic phrases.
            normalized_question = " ".join(question.lower().split())
            normalized_title = " ".join(title.lower().split())

            if normalized_question in normalized_title:
                score += 10

            if score > 0:
                matches.append({
                    "title": title,
                    "content": content,
                    "url": url,
                    "score": score,
                })

        matches.sort(key=lambda item: item["score"], reverse=True)
        matches = matches[:top_k]

        max_score = max(
            (item["score"] for item in matches),
            default=1,
        )

        for item in matches:
            item["relevance"] = round(item["score"] / max_score, 3)
            item.pop("score", None)

        logger.info(
            "Keyword search: database_articles=%s matched_articles=%s",
            len(rows),
            len(matches),
        )

        return matches

    except (sqlite3.Error, FileNotFoundError):
        logger.exception("Database search failed")
        return []


# --------------------------------------------------
# Safe fallback if Gemini is unavailable
# --------------------------------------------------

def evidence_fallback(hits: list[dict]) -> str:
    if not hits:
        return (
            "I could not find sufficient evidence in the knowledge base "
            "to answer this question. Please escalate to a human expert."
        )

    lines = [
        "I found relevant knowledge articles, but AI answer generation "
        "is unavailable right now. Please review these sources:"
    ]

    for index, hit in enumerate(hits[:5], start=1):
        lines.append(
            f"{index}. {hit['title']} — {hit['url']}"
        )

    return "\n".join(lines)


# --------------------------------------------------
# Gemini answer generation
# --------------------------------------------------

def generate_answer(question: str, hits: list[dict]) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    if not api_key:
        logger.warning("GEMINI_API_KEY is not configured")
        return evidence_fallback(hits)

    context_parts = []

    for index, hit in enumerate(hits, start=1):
        content = " ".join(str(hit.get("content", "")).split())
        content = content[:5000]

        context_parts.append(
            f"[{index}] Title: {hit.get('title', '')}\n"
            f"URL: {hit.get('url', '')}\n"
            f"Content: {content}"
        )

    prompt = f"""
You are the Knowledge Brain support assistant.

Answer the user's question using only the source articles provided.
Do not invent facts, settings, or instructions.
If the sources do not contain enough evidence, clearly say so.
Give clear numbered steps when appropriate.
Cite supporting articles with their numbers, such as [1].
Do not claim that an action was completed.

USER QUESTION:
{question}

SOURCE ARTICLES:
{chr(10).join(context_parts)}
"""

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

        answer = (response.text or "").strip()

        if not answer:
            logger.warning("Gemini returned an empty answer")
            return evidence_fallback(hits)

        return answer

    except Exception as exc:
        # Do not log the API key.
        logger.exception(
            "Gemini generation failed: %s",
            type(exc).__name__,
        )
        return evidence_fallback(hits)


# --------------------------------------------------
# Escalation storage
# --------------------------------------------------

def initialize_escalations_table():
    try:
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(str(DATABASE_PATH), timeout=30) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS escalations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    resolution TEXT,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                )
            """)
    except sqlite3.Error:
        logger.exception("Could not initialize escalation table")


def create_escalation(question: str, reason: str) -> int:
    initialize_escalations_table()

    with sqlite3.connect(str(DATABASE_PATH), timeout=30) as connection:
        cursor = connection.execute(
            """
            INSERT INTO escalations
                (question, reason, status, created_at)
            VALUES (?, ?, 'open', ?)
            """,
            (
                question,
                reason,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cursor.lastrowid


# --------------------------------------------------
# Ingestion
# --------------------------------------------------

def run_ingestion_job():
    global ingestion_running, ingestion_last_result

    try:
        if crawl is None:
            raise RuntimeError(
                "Could not import crawl from ingestion.crawler. "
                "Check the crawler function name."
            )

        result = crawl()
        ingestion_last_result = (
            str(result) if result is not None else "completed"
        )
        logger.info("Ingestion finished: %s", ingestion_last_result)

    except Exception as exc:
        ingestion_last_result = f"failed: {type(exc).__name__}"
        logger.exception("Ingestion failed")

    finally:
        with ingestion_lock:
            ingestion_running = False


# --------------------------------------------------
# API routes
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "AI Tech Consultant — Knowledge Brain",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    database_exists = DATABASE_PATH.exists()
    total_articles = 0

    if database_exists:
        try:
            with get_connection() as connection:
                table = connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='articles'
                    """
                ).fetchone()

                if table:
                    total_articles = connection.execute(
                        "SELECT COUNT(*) FROM articles"
                    ).fetchone()[0]
        except sqlite3.Error:
            logger.exception("Health check database query failed")

    with ingestion_lock:
        running = ingestion_running

    return {
        "status": "ok",
        "database_exists": database_exists,
        "database_path": str(DATABASE_PATH),
        "total_articles": total_articles,
        "llm_provider": "gemini",
        "llm_configured": bool(
            os.getenv("GEMINI_API_KEY", "").strip()
        ),
        "retrieval": "sqlite_keyword_search",
        "ingestion_running": running,
        "ingestion_last_result": ingestion_last_result,
    }


@app.get("/ingestion/status")
def ingestion_status():
    database_exists = DATABASE_PATH.exists()
    total_articles = 0
    added = 0

    if database_exists:
        try:
            with get_connection() as connection:
                table = connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='articles'
                    """
                ).fetchone()

                if table:
                    total_articles = connection.execute(
                        "SELECT COUNT(*) FROM articles"
                    ).fetchone()[0]
        except sqlite3.Error:
            logger.exception("Could not read ingestion status")

    with ingestion_lock:
        running = ingestion_running
        last_result = ingestion_last_result

    return {
        "total_articles": total_articles,
        "ingestion_status": {"added": added},
        "database_exists": database_exists,
        "job_running": running,
        "last_result": last_result,
    }


@app.post("/ingestion/run")
def start_ingestion(background_tasks: BackgroundTasks):
    global ingestion_running, ingestion_last_result

    with ingestion_lock:
        if ingestion_running:
            return {
                "status": "already_running",
                "message": "Ingestion is already running.",
            }

        ingestion_running = True
        ingestion_last_result = "running"

    background_tasks.add_task(run_ingestion_job)

    return {
        "status": "started",
        "message": "Ingestion started. Check /ingestion/status for progress.",
    }


@app.post("/query")
def query_knowledge(request: QueryRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    hits = search_articles(question, request.top_k)

    public_sources = [
        {
            "title": hit["title"],
            "url": hit["url"],
            "relevance": hit["relevance"],
        }
        for hit in hits
    ]

    if not hits:
        return {
            "answer": (
                "I could not find sufficient evidence in the knowledge "
                "base to answer this question. Please escalate to a human expert."
            ),
            "status": "insufficient_evidence",
            "sources": [],
            "needs_human": True,
        }

    answer = generate_answer(question, hits)

    if answer.startswith(
        "I found relevant knowledge articles, but AI answer generation"
    ):
        return {
            "answer": answer,
            "status": "generation_error",
            "sources": public_sources,
            "needs_human": True,
        }

    return {
        "answer": answer,
        "status": "answered",
        "sources": public_sources,
        "needs_human": False,
    }


@app.get("/escalations")
def list_escalations():
    if not DATABASE_PATH.exists():
        return {"escalations": []}

    initialize_escalations_table()

    with sqlite3.connect(str(DATABASE_PATH), timeout=30) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM escalations ORDER BY id DESC"
        ).fetchall()

    return {"escalations": [dict(row) for row in rows]}


@app.post("/escalations")
def create_escalation_endpoint(request: EscalationRequest):
    if not DATABASE_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Knowledge database is not available.",
        )

    try:
        escalation_id = create_escalation(
            request.question.strip(),
            request.reason.strip(),
        )
        return {
            "status": "created",
            "id": escalation_id,
        }
    except sqlite3.Error:
        logger.exception("Could not create escalation")
        raise HTTPException(
            status_code=500,
            detail="Could not create escalation.",
        )


@app.post("/escalations/{escalation_id}/resolve")
def resolve_escalation(escalation_id: int, resolution: str):
    if not DATABASE_PATH.exists():
        raise HTTPException(status_code=404, detail="Escalation not found.")

    initialize_escalations_table()

    with sqlite3.connect(str(DATABASE_PATH), timeout=30) as connection:
        cursor = connection.execute(
            """
            UPDATE escalations
            SET status='resolved',
                resolution=?,
                resolved_at=?
            WHERE id=?
            """,
            (
                resolution,
                datetime.now(timezone.utc).isoformat(),
                escalation_id,
            ),
        )

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="Escalation not found.",
            )

    return {
        "status": "resolved",
        "id": escalation_id,
    }
