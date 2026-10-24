from app.pipeline.claim_verifier import NLIResult
from app.pipeline.uncertainty import (
    CONTRADICTED, SUPPORTED, UNVERIFIED, Thresholds, decide,
)


def nli(entail=0.0, contra=0.0):
    return NLIResult(entailment=entail, contradiction=contra, neutral=1 - entail - contra)


def test_strong_entailment_is_supported():
    v = decide([nli(entail=0.95), nli(entail=0.1)])
    assert v.label == SUPPORTED
    assert v.evidence_index == 0


def test_strong_contradiction_is_contradicted():
    v = decide([nli(entail=0.05), nli(contra=0.92)])
    assert v.label == CONTRADICTED
    assert v.evidence_index == 1


def test_weak_signals_are_unverified():
    v = decide([nli(entail=0.4, contra=0.3)])
    assert v.label == UNVERIFIED
    assert v.evidence_index is None


def test_conflicting_evidence_is_unverified():
    v = decide([nli(entail=0.9), nli(contra=0.9)])
    assert v.label == UNVERIFIED
    assert "conflict" in v.reason.lower()


def test_no_evidence_is_unverified():
    assert decide([]).label == UNVERIFIED


def test_thresholds_change_the_outcome():
    results = [nli(contra=0.76)]
    assert decide(results, Thresholds(contradict=0.9)).label == UNVERIFIED
    assert decide(results, Thresholds(contradict=0.7)).label == CONTRADICTED