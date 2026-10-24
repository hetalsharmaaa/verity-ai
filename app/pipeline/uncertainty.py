"""Uncertainty handling: turn raw NLI probabilities into a final verdict.

Rules (deliberately conservative):
  - supported:    some passage strongly entails the claim, none strongly contradicts it
  - contradicted: some passage strongly contradicts the claim, none strongly entails it
  - unverified:   everything else, including weak signals and conflicting evidence

Unverified does NOT mean false. It means the available evidence is not enough.
"""

from dataclasses import dataclass

from app.pipeline.claim_verifier import NLIResult

SUPPORTED = "supported"
CONTRADICTED = "contradicted"
UNVERIFIED = "unverified"


@dataclass
class Thresholds:
    support: float = 0.80      # min entailment probability to call a claim supported
    contradict: float = 0.80   # min contradiction probability to call it contradicted


@dataclass
class Verdict:
    label: str                 # supported / contradicted / unverified
    score: float               # raw NLI probability behind the verdict (NOT calibrated)
    reason: str                # human-readable explanation
    evidence_index: int | None # index of the passage that drove the verdict, if any


def decide(results: list[NLIResult], thresholds: Thresholds | None = None) -> Verdict:
    """Combine NLI results for one claim (one result per retrieved passage)."""
    t = thresholds or Thresholds()

    if not results:
        return Verdict(UNVERIFIED, 0.0, "No evidence passages were available to check this claim.", None)

    best_support = max(range(len(results)), key=lambda i: results[i].entailment)
    best_contra = max(range(len(results)), key=lambda i: results[i].contradiction)
    support = results[best_support].entailment
    contra = results[best_contra].contradiction

    strong_support = support >= t.support
    strong_contra = contra >= t.contradict

    if strong_support and strong_contra:
        return Verdict(
            UNVERIFIED,
            max(support, contra),
            "Conflicting evidence: one passage supports this claim and another contradicts it. "
            "A human should review it.",
            None,
        )
    if strong_support:
        return Verdict(SUPPORTED, support, "A source passage supports this claim.", best_support)
    if strong_contra:
        return Verdict(CONTRADICTED, contra, "A source passage contradicts this claim.", best_contra)

    return Verdict(
        UNVERIFIED,
        max(support, contra),
        f"No passage clearly supports or contradicts this claim "
        f"(best support {support:.2f}, best contradiction {contra:.2f}). "
        "Missing evidence does not mean the claim is false.",
        None,
    )