import json

import numpy as np
import pandas as pd
import pytest

from newsmood import ingest, pipeline, topics
from newsmood.models import HashingEncoder, LexiconSentiment
from newsmood.synthetic import generate

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>Floods &amp; landslides kill 12 in <b>Kerala</b></title><link>https://x.org/a?utm=1</link>
<description>&lt;p&gt;Rescue teams are searching.&lt;/p&gt;</description><pubDate>Thu, 05 Nov 2026 10:00:00 GMT</pubDate></item>
<item><title>Short</title><link>https://x.org/b</link></item>
</channel></rss>"""


def test_parse_feed_cleans_html_and_skips_short_titles():
    rows = ingest.parse_feed("Test", RSS, pd.Timestamp.now(tz="UTC").to_pydatetime())
    assert len(rows) == 1
    assert rows[0]["title"] == "Floods & landslides kill 12 in Kerala"
    assert rows[0]["summary"] == "Rescue teams are searching."
    assert rows[0]["published"].year == 2026


def test_article_id_ignores_tracking_params():
    assert ingest.article_id("https://x.org/a?utm=1", "t") == ingest.article_id("https://x.org/a/", "other")


def test_merge_history_dedupes_and_returns_only_new():
    a = generate(hours=24, seed=1)
    hist, fresh = ingest.merge_history(None, a)
    assert len(fresh) == len(hist) == a["id"].nunique()
    hist2, fresh2 = ingest.merge_history(hist, a)
    assert len(fresh2) == 0 and len(hist2) == len(hist)


def test_lexicon_probabilities_are_valid():
    p = LexiconSentiment().predict(["Missile attack kills dozens", "Peace deal brings hope", "Council meets on Tuesday"])
    assert np.allclose(p.sum(1), 1)
    assert p[0].argmax() == 0 and p[1].argmax() == 2 and p[2].argmax() == 1


def test_topics_recover_planted_stories():
    df = generate(hours=48, seed=3)
    texts = pipeline.text_for(df)
    res = topics.discover(texts, HashingEncoder().encode(texts), min_size=5)
    assert len(res["keywords"]) >= 5
    assert res["silhouette"] is not None and res["silhouette"] > 0.2
    words = " ".join(w for kws in res["keywords"].values() for w in kws)
    assert "malaria" in words or "vaccine" in words


def test_pipeline_end_to_end_and_rescoring(tmp_path):
    hist_path, out = tmp_path / "a.parquet", tmp_path / "site"
    articles = generate(hours=48)
    r1 = pipeline.run(lightweight=True, articles=articles, out_dir=out, history_path=hist_path)
    assert r1["new"] == r1["history"] > 0
    r2 = pipeline.run(lightweight=True, articles=articles, out_dir=out, history_path=hist_path)
    assert r2["new"] == 0 and r2["history"] == r1["history"]
    hist = pd.read_parquet(hist_path)
    assert hist["tone"].notna().all() and hist["tone"].between(-1, 1).all()
    payload = json.loads((out / "dashboard.json").read_text())
    assert payload["stories"] and payload["points"] and payload["sources"]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    os.environ["LIGHTWEIGHT_MODELS"] = "1"
    from fastapi.testclient import TestClient

    from api import main

    out = tmp_path_factory.mktemp("site")
    pipeline.run(lightweight=True, articles=generate(), out_dir=out, history_path=out / "h.parquet")
    main.SITE_DATA_DIR = out
    with TestClient(main.app) as c:
        yield c


def test_api_analyze(client):
    r = client.post("/analyze", json={"text": "Scientists celebrate malaria vaccine breakthrough"})
    body = r.json()
    assert r.status_code == 200 and body["label"] == "positive"
    story = body["closest_story"]
    assert story is not None and story["similarity"] > 0.2
    assert any(w in story["name"].lower() for w in ("malaria", "vaccine", "scientists"))
    assert client.post("/analyze", json={"text": ""}).status_code == 422
