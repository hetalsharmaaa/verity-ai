import pytest

from app.pipeline.evidence_retriever import EvidenceRetriever, build_passages

SOURCE = (
    "Paris is the capital of France. The Eiffel Tower was completed in 1889. "
    "Berlin is the capital of Germany. The Brandenburg Gate is in Berlin. "
    "Tokyo is the capital of Japan."
)


@pytest.fixture(scope="module")
def retriever():
    r = EvidenceRetriever()  # loads the model once for all tests in this file
    r.index(SOURCE)
    return r


def test_passages_offsets_match_source():
    for p in build_passages(SOURCE):
        assert SOURCE[p.start:p.end] == p.text


def test_empty_source_gives_no_passages():
    assert build_passages("   ") == []


def test_retrieves_relevant_passage(retriever):
    results = retriever.search("The capital of Germany is Berlin.", top_k=1)
    assert "Germany" in results[0].passage.text


def test_results_sorted_by_score(retriever):
    results = retriever.search("When was the Eiffel Tower built?", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)