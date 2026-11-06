"""Generated headlines (offline runs and tests): several multi-source stories plus noise."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEEDS
from .ingest import COLUMNS, article_id

STORIES = [
    ("Ceasefire talks resume as mediators push for {x} agreement in Gaza", ["new", "fragile", "lasting", "historic"]),
    ("Central bank warns of {x} inflation as rate decision looms", ["persistent", "rising", "sticky", "global"]),
    ("Floods kill dozens and displace thousands in {x}", ["Bangladesh", "southern Brazil", "northern India", "Kenya"]),
    ("Climate summit delegates reach {x} deal on fossil fuel transition", ["breakthrough", "landmark", "partial", "historic"]),
    ("Election results show {x} lead for opposition coalition", ["narrow", "surprise", "commanding", "early"]),
    ("Scientists celebrate {x} breakthrough in malaria vaccine trial", ["record", "major", "promising", "historic"]),
    ("Missile strike on {x} kills civilians, officials say", ["Kharkiv", "Odesa", "power grid", "residential block"]),
    ("Tech giants face {x} antitrust ruling in Europe", ["landmark", "record", "sweeping", "tough"]),
]
NOISE = ["Museum reopens after renovation", "Local team wins regional cup final", "Train services disrupted by strike",
         "Hospital opens new cancer wing", "Wildfire forces evacuations in hills", "Startup raises funding for battery plant",
         "Parliament debates new data law", "Heatwave expected to break records", "Port workers announce new strike"]


def generate(hours: int = 72, seed: int = 5, end: pd.Timestamp | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = end or pd.Timestamp.now(tz="UTC").floor("min")
    sources = list(FEEDS)
    rows = []
    for template, fills in STORIES:
        for _ in range(rng.integers(8, 16)):
            title = template.format(x=rng.choice(fills))
            src = rng.choice(sources)
            rows.append((title, src))
    for _ in range(40):
        rows.append((f"{rng.choice(NOISE)} in {rng.choice(['Lyon', 'Pune', 'Leeds', 'Osaka', 'Austin', 'Nairobi'])}", rng.choice(sources)))
    out = []
    for i, (title, src) in enumerate(rows):
        t = end - pd.Timedelta(minutes=int(rng.integers(0, hours * 60)))
        link = f"https://example.org/{src.replace(' ', '').lower()}/{i}"
        out.append({"id": article_id(link, title), "source": src, "title": title, "summary": f"{title}. More details to follow.",
                    "link": link, "published": t, "fetched_at": end})
    return pd.DataFrame(out, columns=COLUMNS)
