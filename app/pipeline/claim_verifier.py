"""Claim verification: does a passage support, contradict, or say nothing
about a claim? Uses a pretrained NLI (natural language inference) model.

This module returns raw probabilities only. Turning them into a final label
(supported / contradicted / unverified) happens in a later step.
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_NLI_MODEL = "cross-encoder/nli-distilroberta-base"

# Documented output order for the cross-encoder NLI models. Only used as a
# fallback if the model config has generic names like "LABEL_0".
_FALLBACK_ORDER = ["contradiction", "entailment", "neutral"]
_REQUIRED_LABELS = {"entailment", "contradiction", "neutral"}


@dataclass
class NLIResult:
    entailment: float     # probability the passage supports the claim
    contradiction: float  # probability the passage conflicts with the claim
    neutral: float        # probability the passage says nothing either way


class ClaimVerifier:
    def __init__(self, model_name: str = DEFAULT_NLI_MODEL):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # inference mode: no training behaviour
        self._label_index = self._build_label_index()

    def _build_label_index(self) -> dict[str, int]:
        """Map 'entailment' etc. to the model's output positions."""
        mapping = {
            str(label).lower(): int(idx)
            for idx, label in self.model.config.id2label.items()
        }
        if not _REQUIRED_LABELS.issubset(mapping):
            mapping = {label: i for i, label in enumerate(_FALLBACK_ORDER)}
        return mapping

    def verify_batch(self, premises: list[str], claim: str) -> list[NLIResult]:
        """Check one claim against several passages in a single model call."""
        if not premises:
            return []
        inputs = self.tokenizer(
            premises,
            [claim] * len(premises),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():  # we are not training, so skip gradient tracking
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)  # logits -> probabilities

        results = []
        for row in probs:
            results.append(
                NLIResult(
                    entailment=float(row[self._label_index["entailment"]]),
                    contradiction=float(row[self._label_index["contradiction"]]),
                    neutral=float(row[self._label_index["neutral"]]),
                )
            )
        return results

    def verify(self, premise: str, claim: str) -> NLIResult:
        return self.verify_batch([premise], claim)[0]


if __name__ == "__main__":
    verifier = ClaimVerifier()
    print("Label mapping:", verifier._label_index)

    cases = [
        ("Patients receiving the treatment recovered 12% faster than the control group.",
         "Patients recovered faster with the treatment."),
        ("Patients receiving the treatment recovered 12% faster than the control group.",
         "The treatment improved recovery rates by 47%."),
        ("Patients receiving the treatment recovered 12% faster than the control group.",
         "The treatment was developed in Japan."),
        ("The trial enrolled 1,200 patients across six hospitals in 2022.",
         "The study included 1,200 patients."),
        ("The trial enrolled 1,200 patients across six hospitals in 2022.",
         "The trial enrolled 5,000 patients."),
    ]
    for premise, claim in cases:
        r = verifier.verify(premise, claim)
        print(f"\nPREMISE: {premise}\nCLAIM:   {claim}")
        print(f"  entail={r.entailment:.3f}  contra={r.contradiction:.3f}  neutral={r.neutral:.3f}")