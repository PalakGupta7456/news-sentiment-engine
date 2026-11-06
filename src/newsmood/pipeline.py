"""Hourly run: ingest -> score new articles -> discover topics -> publish."""
from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from . import ingest, topics
from .config import HISTORY, MIN_TOPIC_SIZE, SITE_DATA_DIR, TOPIC_WINDOW_HOURS
from .models import LABELS, load

log = logging.getLogger("newsmood")


def text_for(df: pd.DataFrame) -> list[str]:
    # Headline carries most of the signal; the first sentence of the summary adds context.
    return (df["title"] + ". " + df["summary"].fillna("").str.split(r"(?<=[.!?])\s", n=1, regex=True).str[0]).tolist()


def score(df: pd.DataFrame, sentiment) -> pd.DataFrame:
    p = sentiment.predict(text_for(df))
    return df.assign(p_neg=p[:, 0].round(4), p_neu=p[:, 1].round(4), p_pos=p[:, 2].round(4),
                     label=[LABELS[i] for i in p.argmax(1)], tone=(p[:, 2] - p[:, 0]).round(4))


def build_payload(hist: pd.DataFrame, topic: dict, recent: pd.DataFrame, feed_status: dict, timings: dict, model_names: dict) -> dict:
    hist = hist.assign(published=pd.to_datetime(hist["published"], utc=True))
    day = hist.assign(date=hist["published"].dt.strftime("%Y-%m-%d"))
    daily = day.groupby("date").agg(n=("id", "size"), tone=("tone", "mean"), neg=("label", lambda s: (s == "negative").mean())).tail(30)
    by_source = (hist[hist["published"] >= hist["published"].max() - pd.Timedelta(days=7)]
                 .groupby("source").agg(n=("id", "size"), tone=("tone", "mean"),
                                         neg=("label", lambda s: (s == "negative").mean()),
                                         pos=("label", lambda s: (s == "positive").mean()))
                 .sort_values("tone"))

    labels, kw = topic["labels"], topic["keywords"]
    recent = recent.assign(topic=labels, x=topic["coords"][:, 0].round(3), y=topic["coords"][:, 1].round(3))
    stories = []
    for t, g in recent[recent["topic"] >= 0].groupby("topic"):
        g = g.sort_values("published", ascending=False)
        stories.append({
            "id": int(t), "name": topics.title_case(kw.get(t, [])), "keywords": kw.get(t, []), "n": int(len(g)),
            "sources": int(g["source"].nunique()), "tone": round(float(g["tone"].mean()), 3),
            "neg": round(float((g["label"] == "negative").mean()), 3),
            "headlines": [{"t": r.title, "s": r.source, "u": r.link, "tone": round(float(r.tone), 2)} for r in g.head(6).itertuples()],
        })
    stories.sort(key=lambda s: (-s["sources"], -s["n"]))
    rank = {s["id"]: i for i, s in enumerate(stories)}
    points = [{"x": float(r.x), "y": float(r.y), "k": int(rank.get(r.topic, -1)), "tone": round(float(r.tone), 2), "t": r.title, "s": r.source}
              for r in recent.itertuples()]
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "window_hours": TOPIC_WINDOW_HOURS,
        "totals": {"articles_30d": int(len(hist)), "articles_window": int(len(recent)), "sources": int(hist["source"].nunique())},
        "overall": {"tone": round(float(recent["tone"].mean()), 3) if len(recent) else 0,
                    "mix": {lab: round(float((recent["label"] == lab).mean()), 3) for lab in LABELS} if len(recent) else {}},
        "daily": {"date": daily.index.tolist(), "n": daily["n"].astype(int).tolist(),
                  "tone": daily["tone"].round(3).tolist(), "neg": daily["neg"].round(3).tolist()},
        "sources": [{"name": s, **{k: round(float(v), 3) if k != "n" else int(v) for k, v in row.items()}} for s, row in by_source.iterrows()],
        "stories": stories[:12],
        "points": points,
        "quality": {"silhouette": None if topic["silhouette"] is None else round(topic["silhouette"], 3),
                    "coverage": round(topic["coverage"], 3), "n_topics": len(stories)},
        "feeds": feed_status, "timings": timings, "models": model_names,
    }


def run(lightweight: bool = False, articles: pd.DataFrame | None = None, out_dir=SITE_DATA_DIR, history_path=HISTORY) -> dict:
    timings = {}
    t0 = time.time()
    if articles is None:
        articles, feed_status = ingest.fetch_all()
    else:
        feed_status = articles.groupby("source").size().astype(int).to_dict()
    timings["ingest_s"] = round(time.time() - t0, 1)
    old = pd.read_parquet(history_path) if history_path.exists() else None
    hist, fresh = ingest.merge_history(old, articles)
    log.info("fetched %d, new %d, history %d", len(articles), len(fresh), len(hist))

    t0 = time.time()
    sentiment, encoder = load(lightweight)
    timings["load_models_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    # Score every article that has no score yet: new arrivals, or everything after a model change.
    cols = ["p_neg", "p_neu", "p_pos", "label", "tone"]
    for c in cols:
        if c not in hist:
            hist[c] = np.nan if c != "label" else None
    missing = hist["tone"].isna()
    if missing.any():
        scored = score(hist[missing], sentiment)
        hist.loc[missing, cols] = scored[cols].to_numpy()
    hist[["p_neg", "p_neu", "p_pos", "tone"]] = hist[["p_neg", "p_neu", "p_pos", "tone"]].astype(float)
    timings["sentiment_s"] = round(time.time() - t0, 1)

    t0 = time.time()
    pub = pd.to_datetime(hist["published"], utc=True)
    recent = hist[pub >= pub.max() - pd.Timedelta(hours=TOPIC_WINDOW_HOURS)].sort_values("published").reset_index(drop=True)
    emb = encoder.encode(text_for(recent)) if len(recent) else np.zeros((0, 2))
    topic = topics.discover(text_for(recent), emb, MIN_TOPIC_SIZE)
    timings["topics_s"] = round(time.time() - t0, 1)

    history_path.parent.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(history_path, index=False)
    names = {"sentiment": type(sentiment).__name__, "encoder": type(encoder).__name__}
    if not lightweight:
        from .config import EMBED_MODEL, SENTIMENT_MODEL
        names = {"sentiment": SENTIMENT_MODEL, "encoder": EMBED_MODEL}
    payload = build_payload(hist, topic, recent, feed_status, timings, names)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dashboard.json").write_text(json.dumps(payload, separators=(",", ":"), default=str))
    return {"fetched": len(articles), "new": len(fresh), "history": len(hist), **payload["quality"], "timings": timings}


def main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lightweight", action="store_true", help="lexicon + hashing models (no downloads)")
    ap.add_argument("--synthetic", action="store_true", help="use generated headlines instead of live feeds")
    a = ap.parse_args()
    articles = None
    if a.synthetic:
        from .synthetic import generate
        articles = generate()
    print(json.dumps(run(lightweight=a.lightweight, articles=articles), indent=2))


if __name__ == "__main__":
    main()
