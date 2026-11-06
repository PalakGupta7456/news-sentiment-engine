"""Sources, models and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
HISTORY = DATA_DIR / "articles.parquet"
SITE_DATA_DIR = ROOT / "docs" / "data"

# Public RSS feeds (headline + summary only)
FEEDS = {
    "BBC": "https://feeds.bbci.co.uk/news/world/rss.xml",
    "The Guardian": "https://www.theguardian.com/world/rss",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "NPR": "https://feeds.npr.org/1004/rss.xml",
    "The Hindu": "https://www.thehindu.com/news/international/feeder/default.rss",
    "Indian Express": "https://indianexpress.com/section/world/feed/",
    "DW": "https://rss.dw.com/rdf/rss-en-world",
    "France 24": "https://www.france24.com/en/rss",
    "CBC": "https://www.cbc.ca/webfeed/rss/rss-world",
    "ABC News (AU)": "https://www.abc.net.au/news/feed/51120/rss.xml",
}

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"   # negative / neutral / positive
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"                 # 384-d sentence embeddings
TOPIC_WINDOW_HOURS = 48
HISTORY_DAYS = 30
MIN_TOPIC_SIZE = 5
