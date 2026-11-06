"""FastAPI: score any headline and match it to today's stories.

    POST /analyze {"text": "..."}  ->  sentiment probabilities + closest live story
Models load once at startup (baked into the Docker image at build time).
"""
from __future__ import annotations

import json
import os
import urllib.request
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from newsmood import __version__
from newsmood.config import SITE_DATA_DIR
from newsmood.models import LABELS, load

STATE: dict = {}
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "https://palakgupta7456.github.io/news-sentiment-engine/data/dashboard.json")


def _stories() -> list[dict]:
    local = SITE_DATA_DIR / "dashboard.json"
    try:
        raw = local.read_text() if local.exists() else urllib.request.urlopen(DASHBOARD_URL, timeout=10).read()
        return json.loads(raw)["stories"]
    except Exception:  # noqa: BLE001 - the API still scores text without live stories
        return []


@asynccontextmanager
async def lifespan(_: FastAPI):
    lightweight = os.getenv("LIGHTWEIGHT_MODELS", "0") == "1"
    STATE["sentiment"], STATE["encoder"] = load(lightweight)
    stories = _stories()
    STATE["stories"] = stories
    if stories:
        texts = [" ".join(h["t"] for h in s["headlines"][:4]) for s in stories]
        STATE["story_emb"] = STATE["encoder"].encode(texts)
    yield


app = FastAPI(title="News Sentiment Engine API", version=__version__, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


class AnalyzeIn(BaseModel):
    text: str = Field(min_length=3, max_length=2000, examples=["Central bank holds rates as inflation cools"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": "sentiment" in STATE, "live_stories": len(STATE.get("stories", [])), "version": __version__}


@app.post("/analyze")
def analyze(body: AnalyzeIn) -> dict:
    if "sentiment" not in STATE:
        raise HTTPException(503, "Models not loaded")
    p = STATE["sentiment"].predict([body.text])[0]
    out = {"text": body.text, "label": LABELS[int(p.argmax())], "tone": round(float(p[2] - p[0]), 4),
           "probabilities": {lab: round(float(v), 4) for lab, v in zip(LABELS, p, strict=True)}, "closest_story": None}
    if "story_emb" in STATE:
        sims = STATE["story_emb"] @ STATE["encoder"].encode([body.text])[0]
        i = int(np.argmax(sims))
        s = STATE["stories"][i]
        out["closest_story"] = {"name": s["name"], "similarity": round(float(sims[i]), 3), "articles": s["n"], "tone": s["tone"]}
    return out


@app.get("/stories")
def stories() -> list[dict]:
    return [{k: s[k] for k in ("name", "keywords", "n", "sources", "tone", "neg")} for s in STATE.get("stories", [])]
