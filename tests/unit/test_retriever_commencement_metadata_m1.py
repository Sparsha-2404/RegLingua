import json

from services.retrieval.deterministic_retriever import DeterministicRetriever


def _write_corpus(tmp_path, entry):
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps([entry]), encoding="utf-8")
    return corpus_path


def _sample_entry():
    return {
        "source_id": "DPDP_TEST_001",
        "regulator": "DPDP",
        "document_title": "DPDP Rules Test",
        "section": "Rule 4",
        "effective_date": "2025-11-14",
        "text": "Consent manager registration conditions and commencement.",
        "source_url": "https://example.com/dpdp-rules",
        "topic": "consent manager registration",
    }


def test_retriever_passes_through_commencement_metadata(tmp_path):
    entry = _sample_entry()
    entry.update({
        "publication_date": "2025-11-14",
        "commencement_date": "2026-11-14",
        "commencement_note": "Scheduled commencement under the applicable rule.",
    })

    retriever = DeterministicRetriever(_write_corpus(tmp_path, entry))
    results = retriever.retrieve(
        query_text="consent manager registration",
        regulator_candidates=["DPDP"],
    )

    assert len(results) == 1
    assert results[0].publication_date == "2025-11-14"
    assert results[0].commencement_date == "2026-11-14"
    assert results[0].commencement_note == (
        "Scheduled commencement under the applicable rule."
    )


def test_retriever_supports_entries_without_commencement_metadata(tmp_path):
    entry = _sample_entry()
    retriever = DeterministicRetriever(_write_corpus(tmp_path, entry))
    results = retriever.retrieve(
        query_text="consent manager registration",
        regulator_candidates=["DPDP"],
    )

    assert len(results) == 1
    assert results[0].publication_date is None
    assert results[0].commencement_date is None
    assert results[0].commencement_note is None
