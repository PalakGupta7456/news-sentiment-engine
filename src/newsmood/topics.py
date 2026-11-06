"""Topic discovery on sentence embeddings, BERTopic-style but transparent:

1. HDBSCAN over L2-normalised embeddings finds dense groups of headlines
   about the same story (no need to choose the number of topics; outliers
   are allowed and labelled -1).
2. Class-based TF-IDF (c-TF-IDF) names each cluster by the words that are
   frequent inside it but rare in the other clusters.
3. PCA projects embeddings to 2-D for the story map.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import silhouette_score

EXTRA_STOP = {"says", "said", "new", "news", "live", "update", "updates", "year", "years", "day", "week", "just", "amid",
              "people", "video", "watch", "latest", "time", "make", "set", "world", "report", "reports"}
STOP = list(ENGLISH_STOP_WORDS | EXTRA_STOP)


def cluster(emb: np.ndarray, min_size: int) -> np.ndarray:
    if len(emb) < min_size * 2:
        return np.full(len(emb), -1)
    return HDBSCAN(min_cluster_size=min_size, min_samples=max(2, min_size // 2), metric="euclidean",
                   cluster_selection_method="leaf", copy=True).fit_predict(emb)


def ctfidf_keywords(docs: list[str], labels: np.ndarray, top_n: int = 5) -> dict[int, list[str]]:
    ids = sorted(set(labels) - {-1})
    if not ids:
        return {}
    joined = [" ".join(d for d, lab in zip(docs, labels, strict=True) if lab == t) for t in ids]
    vec = CountVectorizer(stop_words=STOP, ngram_range=(1, 2), min_df=1, token_pattern=r"(?u)\b[A-Za-z][A-Za-z\-]{2,}\b")
    tf = vec.fit_transform(joined).toarray().astype(float)
    tf = tf / np.maximum(tf.sum(axis=1, keepdims=True), 1)
    avg_words = tf.sum() / len(ids) if len(ids) else 1
    idf = np.log(1 + avg_words / np.maximum(tf.sum(axis=0), 1e-12))
    scores = tf * idf
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for row, t in zip(scores, ids, strict=True):
        words = []
        for w in vocab[np.argsort(-row)]:
            if not any(w in kept or kept in w for kept in words):   # skip "gaza" when "gaza ceasefire" is kept, etc.
                words.append(w)
            if len(words) == top_n:
                break
        out[t] = words
    return out


def title_case(words: list[str]) -> str:
    return ", ".join(re.sub(r"\b(\w)", lambda m: m.group(1).upper(), w) for w in words[:3])


def discover(texts: list[str], emb: np.ndarray, min_size: int) -> dict:
    labels = cluster(emb, min_size)
    keywords = ctfidf_keywords(texts, labels)
    coords = PCA(n_components=2, random_state=0).fit_transform(emb) if len(emb) >= 3 else np.zeros((len(emb), 2))
    clustered = labels >= 0
    sil = None
    if len(set(labels[clustered])) >= 2:
        sil = float(silhouette_score(emb[clustered], labels[clustered], metric="cosine"))
    return {"labels": labels, "keywords": keywords, "coords": coords, "silhouette": sil,
            "coverage": float(clustered.mean()) if len(labels) else 0.0}
