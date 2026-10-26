from services.retrieval.hybrid_retriever import HybridRetriever


class FakeKeywordRetriever:
    def retrieve(
        self,
        query_text,
        regulator_candidates=None,
        topic_hint=None,
        top_k=5,
    ):
        from agents.research_contracts import RetrievedChunk

        return [
            RetrievedChunk(
                source_id="SEBI_IA_002",
                regulator="SEBI",
                document_title="SEBI Advertisement Code",
                section="Clause 3",
                effective_date="2020-07-23",
                text="Guaranteed returns are prohibited.",
                source_url="https://example.com/sebi",
                topic="guaranteed returns marketing language",
                score=0.9,
            ),
            RetrievedChunk(
                source_id="SEBI_MF_003",
                regulator="SEBI",
                document_title="SEBI Mutual Fund Regulations",
                section="Reg. 29",
                effective_date="1996-12-09",
                text="Risk disclosure is required.",
                source_url="https://example.com/mf",
                topic="risk disclosure past performance",
                score=0.3,
            ),
        ]


class FakeVectorIndex:
    def __init__(self):
        self.embeddings = True

    def search(
        self,
        query,
        top_k=5,
        regulator=None,
        topic=None,
        source_id=None,
    ):
        return [
            {
                "source_id": "SEBI_IA_002",
                "regulator": "SEBI",
                "document_title": "SEBI Advertisement Code",
                "section": "Clause 3",
                "effective_date": "2020-07-23",
                "text": "Guaranteed returns are prohibited.",
                "source_url": "https://example.com/sebi",
                "topic": "guaranteed returns marketing language",
                "score": 0.8,
            },
            {
                "source_id": "SEBI_MF_003",
                "regulator": "SEBI",
                "document_title": "SEBI Mutual Fund Regulations",
                "section": "Reg. 29",
                "effective_date": "1996-12-09",
                "text": "Risk disclosure is required.",
                "source_url": "https://example.com/mf",
                "topic": "risk disclosure past performance",
                "score": 0.7,
            },
        ]


def test_hybrid_retriever_combines_keyword_and_vector_scores():
    retriever = HybridRetriever(
        keyword_retriever=FakeKeywordRetriever(),
        vector_index=FakeVectorIndex(),
    )

    results = retriever.retrieve(
        "guaranteed returns",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].source_id == "SEBI_IA_002"


def test_hybrid_retriever_reranks_using_combined_score():
    retriever = HybridRetriever(
        keyword_retriever=FakeKeywordRetriever(),
        vector_index=FakeVectorIndex(),
    )

    results = retriever.retrieve(
        "guaranteed returns",
        top_k=2,
    )

    expected_score = (0.5 * 0.9) + (0.5 * 0.8)

    assert results[0].score == round(expected_score, 4)


def test_hybrid_retriever_preserves_regulatory_metadata():
    retriever = HybridRetriever(
        keyword_retriever=FakeKeywordRetriever(),
        vector_index=FakeVectorIndex(),
    )

    results = retriever.retrieve(
        "guaranteed returns",
        top_k=1,
    )

    result = results[0]

    assert result.regulator == "SEBI"
    assert result.source_id == "SEBI_IA_002"
    assert result.topic == "guaranteed returns marketing language"


def test_hybrid_retriever_rejects_invalid_top_k():
    retriever = HybridRetriever(
        keyword_retriever=FakeKeywordRetriever(),
        vector_index=FakeVectorIndex(),
    )

    try:
        retriever.retrieve("guaranteed returns", top_k=0)
    except ValueError as exc:
        assert "top_k must be >= 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid top_k")


def test_hybrid_retriever_handles_empty_query():
    retriever = HybridRetriever(
        keyword_retriever=FakeKeywordRetriever(),
        vector_index=FakeVectorIndex(),
    )

    results = retriever.retrieve("")

    assert results == []


def test_hybrid_retriever_passes_multiple_regulator_filters():
    class RecordingVectorIndex(FakeVectorIndex):
        def __init__(self):
            super().__init__()
            self.received_regulator = None

        def search(
            self,
            query,
            top_k=5,
            regulator=None,
            topic=None,
            source_id=None,
        ):
            self.received_regulator = regulator
            return super().search(
                query=query,
                top_k=top_k,
                regulator=regulator,
                topic=topic,
                source_id=source_id,
            )

    vector_index = RecordingVectorIndex()
    retriever = HybridRetriever(
        keyword_retriever=FakeKeywordRetriever(),
        vector_index=vector_index,
    )

    retriever.retrieve(
        "guaranteed returns",
        regulator_candidates=["SEBI", "RBI"],
        top_k=2,
    )

    assert vector_index.received_regulator == ["SEBI", "RBI"]
def test_hybrid_retriever_returns_empty_when_both_sources_have_no_results():
    class EmptyKeywordRetriever:
        def retrieve(
            self,
            query_text,
            regulator_candidates=None,
            topic_hint=None,
            top_k=5,
        ):
            return []

    class EmptyVectorIndex:
        embeddings = True

        def search(
            self,
            query,
            top_k=5,
            regulator=None,
            topic=None,
            source_id=None,
        ):
            return []

    retriever = HybridRetriever(
        keyword_retriever=EmptyKeywordRetriever(),
        vector_index=EmptyVectorIndex(),
    )

    assert retriever.retrieve("unsupported query") == []


def test_hybrid_retriever_returns_empty_when_no_regulator_matches():
    class EmptyFilteredKeywordRetriever:
        def retrieve(
            self,
            query_text,
            regulator_candidates=None,
            topic_hint=None,
            top_k=5,
        ):
            return []

    class EmptyFilteredVectorIndex:
        embeddings = True

        def search(
            self,
            query,
            top_k=5,
            regulator=None,
            topic=None,
            source_id=None,
        ):
            assert regulator == ["DPDP"]
            return []

    retriever = HybridRetriever(
        keyword_retriever=EmptyFilteredKeywordRetriever(),
        vector_index=EmptyFilteredVectorIndex(),
    )

    assert retriever.retrieve(
        "guaranteed returns",
        regulator_candidates=["DPDP"],
    ) == []
