from services.embeddings.vector_index import RegulatoryVectorIndex


class FakeEncoder:
    """Small deterministic encoder for fast unit tests."""

    def encode(self, texts):
        vectors = []

        for text in texts:
            text = text.lower()

            if "guaranteed returns" in text:
                vectors.append([1.0, 0.0, 0.0])
            elif "risk disclosure" in text:
                vectors.append([0.0, 1.0, 0.0])
            elif "lending" in text:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([0.1, 0.1, 0.1])

        return vectors

    def encode_one(self, text):
        return self.encode([text])[0]


def test_vector_index_builds_from_corpus():
    index = RegulatoryVectorIndex(encoder=FakeEncoder())

    index.build()

    assert len(index.documents) == 10
    assert index.embeddings.shape == (10, 3)


def test_vector_search_returns_results():
    index = RegulatoryVectorIndex(encoder=FakeEncoder())
    index.build()

    results = index.search("guaranteed returns", top_k=3)

    assert len(results) == 3
    assert results[0]["source_id"] == "SEBI_IA_002"


def test_vector_search_supports_regulator_filter():
    index = RegulatoryVectorIndex(encoder=FakeEncoder())
    index.build()

    results = index.search(
        "guaranteed returns",
        top_k=5,
        regulator="SEBI",
    )

    assert results
    assert all(result["regulator"] == "SEBI" for result in results)


def test_vector_search_rejects_empty_query():
    index = RegulatoryVectorIndex(encoder=FakeEncoder())
    index.build()

    try:
        index.search("")
    except ValueError as exc:
        assert "query must not be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty query")