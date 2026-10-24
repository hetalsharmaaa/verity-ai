"""The full pipeline: claim extraction -> retrieval -> verification -> verdict."""

from dataclasses import dataclass

from app.pipeline.claim_extractor import Claim, extract_claims
from app.pipeline.claim_verifier import ClaimVerifier, NLIResult
from app.pipeline.evidence_retriever import EvidenceRetriever, Passage
from app.pipeline.uncertainty import Thresholds, Verdict, decide


@dataclass
class EvidenceItem:
    passage: Passage
    similarity: float  # how topically close the passage is to the claim
    nli: NLIResult     # what the passage actually says about the claim


@dataclass
class ClaimResult:
    claim: Claim
    verdict: Verdict
    evidence: list[EvidenceItem]


class HallucinationDetector:
    def __init__(
        self,
        retriever: EvidenceRetriever | None = None,
        verifier: ClaimVerifier | None = None,
        thresholds: Thresholds | None = None,
        top_k: int = 3,
    ):
        # Models load once here, then are reused for every request.
        self.retriever = retriever or EvidenceRetriever()
        self.verifier = verifier or ClaimVerifier()
        self.thresholds = thresholds or Thresholds()
        self.top_k = top_k

    def analyze(self, answer: str, source: str) -> list[ClaimResult]:
        """Check every claim in `answer` against the `source` document."""
        self.retriever.index(source)

        results = []
        for claim in extract_claims(answer):
            retrieved = self.retriever.search(claim.text, top_k=self.top_k)
            nli_results = self.verifier.verify_batch(
                [r.passage.text for r in retrieved], claim.text
            )
            evidence = [
                EvidenceItem(passage=r.passage, similarity=r.score, nli=n)
                for r, n in zip(retrieved, nli_results)
            ]
            verdict = decide(nli_results, self.thresholds)
            results.append(ClaimResult(claim=claim, verdict=verdict, evidence=evidence))
        return results


if __name__ == "__main__":
    source = (
        "The trial enrolled 1,200 patients across six hospitals in 2022. "
        "Patients receiving the treatment recovered 12% faster than the control group. "
        "The treatment was well tolerated, with mild nausea as the most common side effect. "
        "The study was funded by a national health grant. "
        "Results were published in 2023."
    )
    answer = (
        "The trial enrolled 1,200 patients across six hospitals. "
        "The treatment improved recovery rates by 47%. "
        "The treatment was developed in Japan. "
        "Patients recovered faster than the control group."
    )

    detector = HallucinationDetector()
    for r in detector.analyze(answer, source):
        print(f"\n[{r.verdict.label.upper()}] (score {r.verdict.score:.2f})  {r.claim.text}")
        print(f"   reason: {r.verdict.reason}")
        if r.verdict.evidence_index is not None:
            print(f"   evidence: {r.evidence[r.verdict.evidence_index].passage.text}")