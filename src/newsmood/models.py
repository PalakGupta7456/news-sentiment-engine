"""Transformer models behind small interfaces, so the pipeline (and tests) can
swap in lightweight stand-ins without touching any analysis code."""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import EMBED_MODEL, SENTIMENT_MODEL

LABELS = ["negative", "neutral", "positive"]


class SentimentModel(Protocol):
    def predict(self, texts: list[str]) -> np.ndarray: ...   # (n, 3) probabilities: neg, neu, pos


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...    # (n, d) L2-normalised embeddings


class HFSentiment:
    """RoBERTa fine-tuned for 3-class sentiment (CardiffNLP, TweetEval)."""

    def __init__(self, name: str = SENTIMENT_MODEL, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        id2label = {i: lab.lower() for i, lab in self.model.config.id2label.items()}
        self.order = [next(i for i, lab in id2label.items() if lab.startswith(want[:3])) for want in LABELS]
        self.batch_size = batch_size

    def predict(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                enc = self.tok(texts[i:i + self.batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt")
                probs = self.model(**enc).logits.softmax(-1).numpy()
                out.append(probs[:, self.order])
        return np.concatenate(out) if out else np.zeros((0, 3))


class HFEncoder:
    def __init__(self, name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)


# ---------- lightweight stand-ins (tests and offline runs) ----------
class LexiconSentiment:
    NEG = {"war", "killed", "dead", "crisis", "attack", "fears", "collapse", "strike", "flood", "crash", "fire", "protest",
           "warns", "violence", "deaths", "inflation", "recession", "conflict", "missile", "earthquake", "scandal", "ban"}
    POS = {"wins", "record", "breakthrough", "agreement", "peace", "growth", "rescue", "celebrates", "boost", "deal",
           "recovery", "award", "launch", "success", "historic", "hope", "cure", "surge", "approves"}

    def predict(self, texts: list[str]) -> np.ndarray:
        out = []
        for t in texts:
            words = set(t.lower().replace(",", " ").split())
            n, p = len(words & self.NEG), len(words & self.POS)
            logits = np.array([n * 1.6, 0.8, p * 1.6])
            e = np.exp(logits - logits.max())
            out.append(e / e.sum())
        return np.array(out).reshape(-1, 3)


class HashingEncoder:
    def __init__(self, dim: int = 512):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vec = HashingVectorizer(n_features=dim, ngram_range=(1, 2), stop_words="english", alternate_sign=False, norm="l2")

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.vec.transform(texts).toarray().astype(np.float32)


def load(lightweight: bool = False) -> tuple[SentimentModel, Encoder]:
    if lightweight:
        return LexiconSentiment(), HashingEncoder()
    return HFSentiment(), HFEncoder()
