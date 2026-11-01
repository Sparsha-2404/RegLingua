"""Loaders for the RegLingua M1 data. Run from the repo root."""
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

def load_labelled(split=None):
    """Labelled statements. The real key is (REGULATOR, ID, LANGUAGE) - ID alone is NOT unique
    because English/Hindi translation pairs share an ID."""
    df = pd.read_csv(DATA / "processed" / "labelled_dataset.csv", dtype=str, keep_default_na=False)
    return df if split is None else df[df.SPLIT == split].reset_index(drop=True)

def load_kb():
    return pd.read_csv(DATA / "regulatory" / "regulatory_kb.csv", dtype=str, keep_default_na=False)

def load_sources():
    return pd.read_csv(DATA / "regulatory" / "regulatory_sources.csv", dtype=str, keep_default_na=False)

def load_retrieval_gold(split="test"):
    g = pd.read_csv(DATA / "evaluation" / "retrieval_gold.csv", dtype=str, keep_default_na=False)
    g["GOLD_CHUNK_LIST"] = g["GOLD_CHUNK_IDS"].str.split("|")
    return g if split is None else g[g.SPLIT == split].reset_index(drop=True)

def recall_at_k(retrieved, gold, k=5):
    """retrieved: list of chunk ids (ranked). gold: list of chunk ids. Hit if any gold id is in top-k."""
    return float(any(c in set(retrieved[:k]) for c in gold))

def parse_effective_date(s):
    """EFFECTIVE_DATE is stored as text 'DD-MM-YYYY' optionally followed by a note, e.g.
    '13-11-2025 (staggered commencement)'. Returns a pandas Timestamp, or NaT for '—'.
    Always parse with dayfirst - never let pandas guess (day/month flips corrupted this column twice)."""
    return pd.to_datetime(str(s)[:10], format="%d-%m-%Y", errors="coerce")
