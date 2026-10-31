from types import SimpleNamespace as NS

from services.retrieval.hybrid_retriever import HybridRetriever


def _chunk(sid, score=0.5, reg="SEBI"):
    return NS(source_id=sid, regulator=reg, document_title="t", section="s",
              effective_date="2020", text="x", source_url="u", topic="", score=score)


class FakeKeyword:
    def __init__(self, results): self.results = results
    def retrieve(self, **kw): return self.results


class FakeIndex:
    embeddings = object()
    def __init__(self, docs): self.docs = docs
    def search(self, **kw):
        return [dict(source_id=d, regulator="SEBI", document_title="t", section="s",
                     effective_date="2020", text="x", source_url="u", topic="",
                     score=s) for d, s in self.docs]


def test_empty_keyword_still_uses_vector(monkeypatch):
    import services.retrieval.hybrid_retriever as h
    monkeypatch.setattr(h, "RetrievedChunk", lambda **kw: NS(**kw))
    r = HybridRetriever(FakeKeyword([]), FakeIndex([("A", 0.9), ("B", 0.5)]))
    out = r.retrieve("हिंदी क्वेरी", regulator_candidates=["SEBI"], top_k=3)
    assert [c.source_id for c in out] == ["A", "B"]


def test_multiple_chunks_same_source_collapse(monkeypatch):
    import services.retrieval.hybrid_retriever as h
    monkeypatch.setattr(h, "RetrievedChunk", lambda **kw: NS(**kw))
    r = HybridRetriever(FakeKeyword([_chunk("A"), _chunk("A"), _chunk("B")]),
                        FakeIndex([("A", 0.9), ("A", 0.8), ("B", 0.1)]))
    out = r.retrieve("q", regulator_candidates=["SEBI"], top_k=5)
    assert [c.source_id for c in out] == ["A", "B"]


def test_both_empty_returns_empty():
    r = HybridRetriever(FakeKeyword([]), FakeIndex([]))
    assert r.retrieve("q", top_k=3) == []
