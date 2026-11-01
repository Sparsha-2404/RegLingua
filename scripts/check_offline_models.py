"""Day 10 (M2): can the demo run fully offline, and what must be cached first?

READ-ONLY: writes nothing, never downloads. HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are forced to 1 for this
process before anything is imported, and the model loader uses local_files_only=True. The commands under
"To enable" are only PRINTED for you to run yourself.

Usage (repo root):  python scripts/check_offline_models.py [--json]
Exit code 0 = fully offline-ready with the embedding model; 1 = demo still runs but on the TF-IDF baseline.
"""
from __future__ import annotations

import os

USER_ENV = dict(os.environ)                 # what the user actually exported, before we force offline below
os.environ["HF_HUB_OFFLINE"] = "1"          # must precede any huggingface_hub / sentence_transformers import
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from services.classifier.model_selection import (  # noqa: E402
    BASELINE_MODEL_PATH, DEFAULT_ENCODER, EMBEDDING_MODEL_PATH, encoder_is_cached, hf_cache_dir, is_offline,
    load_best_available_classifier,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    clf, reason = load_best_available_classifier("embedding")
    kind = reason.split(":", 1)[0]
    cached = encoder_is_cached(DEFAULT_ENCODER)
    todo = []
    if not BASELINE_MODEL_PATH.is_file():
        todo.append("baseline artifact missing: python scripts/train_baseline_classifier.py")
    if not EMBEDDING_MODEL_PATH.is_file():
        todo.append("embedding artifact missing: python scripts/compare_baseline_vs_multilingual.py (writes it + sidecar JSON)")
    if importlib.util.find_spec("sentence_transformers") is None:      # find_spec: no import, no side effects
        todo.append("pip install sentence-transformers")
    if not cached:
        todo.append(f'cache encoder once while ONLINE: python -c "from sentence_transformers import SentenceTransformer as S; S(\'{DEFAULT_ENCODER}\')"')
    if cached and kind != "embedding" and EMBEDDING_MODEL_PATH.is_file():
        todo.append("encoder cache exists but did not load locally (incomplete/corrupt?): see 'reason'; re-cache it while online")
    todo.append("then run the demo with HF_HUB_OFFLINE=1 (Windows: set HF_HUB_OFFLINE=1)")
    report = {"classifier_in_use": kind, "reason": reason, "fully_offline_with_embedding": kind == "embedding",
              "demo_runs_offline": True, "hf_offline_env_set_by_user": is_offline(USER_ENV), "checked_with_local_files_only": True, "hf_cache_dir": str(hf_cache_dir()),
              "encoder_cached": cached, "to_do_before_demo": [] if kind == "embedding" else todo}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"classifier in use : {kind}\nreason            : {reason}")
        print(f"HF_HUB_OFFLINE set by you: {report['hf_offline_env_set_by_user']} (check itself always runs offline) | cache: {report['hf_cache_dir']} | encoder cached: {cached}")
        print("demo offline-safe : YES (TF-IDF baseline needs no network)" + (" and embedding is ready" if kind == "embedding" else ""))
        if todo and kind != "embedding":
            print("To enable the embedding model offline:\n  - " + "\n  - ".join(todo))
    return 0 if kind == "embedding" else 1


if __name__ == "__main__":
    sys.exit(main())
