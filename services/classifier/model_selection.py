"""
DAY 10 (M2): choose the best AVAILABLE risk classifier without ever breaking the deterministic fallback.

load_best_available_classifier(prefer="embedding") -> (classifier, reason)
  * Embedding model is returned only if ALL hold: artifact present, sidecar JSON present and valid,
    library versions compatible (a package recorded in the sidecar but not installed counts as a mismatch),
    a cache snapshot exists, the encoder then ACTUALLY LOADS from local files only, and a smoke
    prediction succeeds. A snapshot directory alone proves nothing; loading + prediction decide.
  * Otherwise the TF-IDF baseline is returned; if that is unusable too, the rule-based MockRiskClassifier.
    `reason` says why and starts with "embedding:", "baseline:" or "mock:".
  * Never raises for a missing/broken optional piece. Never downloads: the default encoder loader is
    SentenceTransformer(name, local_files_only=True) with HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE forced to 1
    for the duration of the load (M3's MultilingualEncoder is NOT used here because it can download).
    The encoding call matches MultilingualEncoder.encode (normalised embeddings), so predictions agree.
All environment access is injectable (env mapping, encoder_factory, cache_probe, versions).
"""
from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator, List, Mapping, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_MODEL_PATH = REPO_ROOT / "models" / "risk_classifier_embedding_logreg_minilm.pkl"
BASELINE_MODEL_PATH = REPO_ROOT / "models" / "risk_classifier_tfidf_logreg.pkl"
DEFAULT_ENCODER = "paraphrase-multilingual-MiniLM-L12-v2"


def is_offline(env: Optional[Mapping[str, str]] = None) -> bool:
    env = os.environ if env is None else env
    return any(str(env.get(k, "")).strip().lower() in ("1", "true", "yes", "on")
               for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))


def hf_cache_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    env = os.environ if env is None else env
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"])
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def encoder_repo_id(name: str) -> str:
    return name if "/" in name else f"sentence-transformers/{name}"


def encoder_is_cached(name: str, env: Optional[Mapping[str, str]] = None) -> bool:
    """True if a snapshot of the encoder exists in the local HF hub cache (no network)."""
    snaps = hf_cache_dir(env) / ("models--" + encoder_repo_id(name).replace("/", "--")) / "snapshots"
    try:
        return snaps.is_dir() and any(p.is_dir() and any(p.iterdir()) for p in snaps.iterdir())
    except OSError:
        return False


def installed_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for pkg in ("scikit-learn", "sentence-transformers"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def version_mismatch(saved: Mapping[str, Any], current: Mapping[str, Any]) -> Optional[str]:
    """Reason string if the artifact cannot be trusted in this environment, else None.

    scikit-learn: must be recorded in the sidecar, installed, and identical (pickle compatibility).
    sentence-transformers: if recorded in the sidecar it must be installed with the same major version
    (not recorded -> nothing to compare, e.g. artifacts built with an injected encoder)."""
    s_sk, c_sk = saved.get("scikit-learn"), current.get("scikit-learn")
    if not s_sk:
        return "scikit-learn version not recorded in sidecar JSON (cannot verify pickle compatibility)"
    if not c_sk:
        return f"scikit-learn {s_sk} recorded in artifact but not installed"
    if s_sk != c_sk:
        return f"scikit-learn {s_sk} (artifact) != {c_sk} (installed)"
    s_st, c_st = saved.get("sentence-transformers"), current.get("sentence-transformers")
    if s_st and not c_st:
        return f"sentence-transformers {s_st} recorded in artifact but not installed"
    if s_st and c_st and str(s_st).split(".")[0] != str(c_st).split(".")[0]:
        return f"sentence-transformers {s_st} (artifact) vs {c_st} (installed): major version differs"
    return None


@contextlib.contextmanager
def force_offline(environ: Optional[Any] = None) -> Iterator[None]:
    """Set HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE to 1 for the duration of the block, then restore."""
    environ = os.environ if environ is None else environ
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    saved = {k: environ.get(k) for k in keys}
    try:
        for k in keys:
            environ[k] = "1"
    except TypeError:  # immutable mapping: rely on local_files_only alone
        yield
        return
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                environ.pop(k, None)
            else:
                environ[k] = v


class LocalOnlyEncoder:
    """SentenceTransformer loaded with local_files_only=True; same encode() output as MultilingualEncoder."""

    def __init__(self, name: str, environ: Optional[Any] = None):
        self.model_name = name
        with force_offline(environ):  # set before the library is imported/instantiated
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(name, local_files_only=True)

    def encode(self, texts) -> List[List[float]]:
        if not texts:
            return []
        return self.model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True).tolist()


def _default_encoder_factory(name: str, environ: Optional[Any] = None):
    return LocalOnlyEncoder(name, environ)   # raises ImportError / OSError when unavailable; never downloads


def _embedding_or_reason(embedding_path, encoder_factory, cache_probe, env, versions) -> Tuple[Optional[Any], str]:
    from services.classifier.embedding_classifier import EmbeddingLogisticClassifier, meta_path_for

    offline = is_offline(env)
    tag = " (HF_HUB_OFFLINE=1)" if offline else ""
    path = Path(embedding_path)
    if not path.is_file():
        return None, f"embedding artifact missing: {path.name}"
    meta_file = meta_path_for(str(path))
    if not meta_file.is_file():
        return None, f"embedding sidecar JSON missing: {meta_file.name}"
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        saved = meta["library_versions"]
        name = meta.get("encoder_name") or DEFAULT_ENCODER
    except Exception as exc:
        return None, f"embedding sidecar JSON unreadable ({type(exc).__name__})"
    bad = version_mismatch(saved, versions())
    if bad:
        return None, f"library version mismatch: {bad}"
    if not cache_probe(name):
        hint = "offline mode, cache is the only source" if offline else "never downloaded automatically"
        return None, f"encoder '{name}' not in local Hugging Face cache{tag}; {hint}; run scripts/check_offline_models.py"
    # A snapshot directory may be incomplete: only a real local-only load + prediction counts below.
    try:
        encoder = encoder_factory(name)
    except ImportError as exc:
        return None, f"encoder import failed (sentence_transformers unavailable?): {exc}"
    except Exception as exc:
        return None, f"encoder could not be loaded from local files{tag} (cache incomplete?): {type(exc).__name__}: {exc}"
    try:
        clf = EmbeddingLogisticClassifier(encoder=encoder).load(str(path))
        pred = clf.predict("smoke test")
        if str(getattr(pred.risk_level, "value", pred.risk_level)) not in ("HIGH", "MEDIUM", "LOW"):
            raise ValueError(f"unexpected smoke prediction {pred.risk_level!r}")
    except Exception as exc:
        return None, f"embedding model failed to load or predict: {type(exc).__name__}: {exc}"
    return clf, f"embedding model available (encoder '{name}' from local cache{tag})"


def load_best_available_classifier(
    prefer: str = "embedding",
    *,
    embedding_path: Path = EMBEDDING_MODEL_PATH,
    baseline_path: Path = BASELINE_MODEL_PATH,
    encoder_factory: Optional[Callable[[str], Any]] = None,
    cache_probe: Optional[Callable[[str], bool]] = None,
    env: Optional[Mapping[str, str]] = None,
    versions: Optional[Callable[[], Mapping[str, Any]]] = None,
) -> Tuple[Any, str]:
    """Return (classifier, reason). Never raises for a missing optional piece."""
    env_ = os.environ if env is None else env
    factory = encoder_factory or (lambda n: _default_encoder_factory(n, env_))
    probe = cache_probe or (lambda n: encoder_is_cached(n, env_))
    reason = "TF-IDF baseline requested"
    if prefer == "embedding":
        try:
            clf, why = _embedding_or_reason(embedding_path, factory, probe, env_, versions or installed_versions)
        except Exception as exc:  # defensive: selection itself must not break the demo
            clf, why = None, f"embedding check failed unexpectedly: {type(exc).__name__}: {exc}"
        if clf is not None:
            return clf, f"embedding: {why}"
        reason = f"embedding unavailable, using baseline: {why}"
    elif prefer != "baseline":
        reason = f"unknown preference '{prefer}', using baseline"
    try:
        from services.classifier.baseline import TfidfLogisticClassifier

        return TfidfLogisticClassifier().load(str(baseline_path)), f"baseline: {reason}"
    except Exception as exc:
        from services.classifier.interface import MockRiskClassifier

        return MockRiskClassifier(), f"mock: {reason}; baseline artifact unusable ({type(exc).__name__}), using rule-based MockRiskClassifier"
