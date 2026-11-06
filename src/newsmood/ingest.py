"""Pull headlines from public RSS feeds and merge them into the article history."""
from __future__ import annotations

import hashlib
import html
import logging
import re
from datetime import UTC, datetime

import feedparser
import pandas as pd
import requests

from .config import FEEDS, HISTORY_DAYS

log = logging.getLogger(__name__)
TAG = re.compile(r"<[^>]+>")
COLUMNS = ["id", "source", "title", "summary", "link", "published", "fetched_at"]


def clean(text: str | None) -> str:
    text = html.unescape(TAG.sub(" ", text or ""))
    return re.sub(r"\s+", " ", text).strip()


def article_id(link: str, title: str) -> str:
    """Stable id: the same story re-published in later pulls is not double counted."""
    key = (link or "").split("?")[0].rstrip("/") or title.lower()
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _published(entry, fallback: datetime) -> datetime:
    for k in ("published_parsed", "updated_parsed"):
        t = entry.get(k)
        if t:
            return datetime(*t[:6], tzinfo=UTC)
    return fallback


def parse_feed(source: str, content: bytes, fetched: datetime) -> list[dict]:
    feed = feedparser.parse(content)
    rows = []
    for e in feed.entries:
        title = clean(e.get("title"))
        if len(title) < 15:
            continue
        rows.append({
            "id": article_id(e.get("link", ""), title), "source": source, "title": title,
            "summary": clean(e.get("summary"))[:600], "link": e.get("link", ""),
            "published": _published(e, fetched), "fetched_at": fetched,
        })
    return rows


def fetch_all(feeds: dict = FEEDS) -> tuple[pd.DataFrame, dict]:
    fetched = datetime.now(UTC)
    rows, status = [], {}
    for source, url in feeds.items():
        try:
            r = requests.get(url, timeout=25, headers={"User-Agent": "news-sentiment-engine/1.0 (portfolio project)"})
            r.raise_for_status()
            items = parse_feed(source, r.content, fetched)
            rows += items
            status[source] = len(items)
        except Exception as exc:  # one broken feed must not stop the pipeline
            log.warning("%s failed: %s", source, exc)
            status[source] = f"error: {type(exc).__name__}"
    return pd.DataFrame(rows, columns=COLUMNS), status


def merge_history(history: pd.DataFrame | None, new: pd.DataFrame, days: int = HISTORY_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (updated history, rows that are genuinely new and need scoring)."""
    if history is None or history.empty:
        fresh = new.drop_duplicates("id")
        merged = fresh
    else:
        fresh = new[~new["id"].isin(history["id"])].drop_duplicates("id")
        merged = pd.concat([history, fresh], ignore_index=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    merged = merged[pd.to_datetime(merged["published"], utc=True) >= cutoff]
    return merged.reset_index(drop=True), fresh.reset_index(drop=True)
