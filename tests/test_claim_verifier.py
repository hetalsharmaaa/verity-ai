import pytest

from app.pipeline.claim_verifier import ClaimVerifier


@pytest.fixture(scope="module")
def verifier():
    return ClaimVerifier()  # load the model once for this file


def test_probabilities_sum_to_one(verifier):
    r = verifier.verify("The cat sat on the mat.", "A cat is on a mat.")
    assert r.entailment + r.contradiction + r.neutral == pytest.approx(1.0, abs=1e-3)


def test_detects_entailment(verifier):
    r = verifier.verify(
        "A man is playing a guitar on stage.",
        "A person is playing a musical instrument.",
    )
    assert r.entailment > r.contradiction
    assert r.entailment > r.neutral


def test_detects_contradiction(verifier):
    r = verifier.verify(
        "The cat is sleeping on the sofa.",
        "The cat is awake and running outside.",
    )
    assert r.contradiction > r.entailment


def test_unrelated_is_not_entailed(verifier):
    r = verifier.verify("The restaurant opened in 2019.", "The mountain is 3,000 meters tall.")
    assert r.entailment < 0.5


def test_batch_matches_input_length(verifier):
    results = verifier.verify_batch(["One sentence here.", "Another sentence here."], "A claim.")
    assert len(results) == 2
    assert verifier.verify_batch([], "A claim.") == []