"""Evidence retrieval: find the source passages most relevant to a claim.

The source document is split into overlapping two-sentence passages. Passages
and claims are embedded with a pretrained model, and passages are ranked by
cosine similarity. A high score means "same topic", NOT "supports the claim".
"""

from dataclasses import dataclass

import torch
from sentence_transformers import SentenceTransformer

# Reuse the sentence splitter from the claim extractor.
from app.pipeline.claim_extractor import _split_segments

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WINDOW_SIZE = 2  # sentences per passage


@dataclass
class Passage:
    id: int
    text: str
    start: int  # character offset in the source document (inclusive)
    end: int    # character offset in the source document (exclusive)


@dataclass
class Evidence:
    passage: Passage
    score: float  # cosine similarity, roughly -1 to 1 (higher = more similar)


def build_passages(source_text: str, window: int = WINDOW_SIZE) -> list[Passage]:
    """Split a source into overlapping windows of consecutive sentences."""
    sentences = []  # (start, end) of each non-empty sentence
    for seg_start, seg_end in _split_segments(source_text):
        raw = source_text[seg_start:seg_end]
        stripped = raw.strip()
        if not stripped:
            continue
        start = seg_start + (len(raw) - len(raw.lstrip()))
        sentences.append((start, start + len(stripped)))

    if not sentences:
        return []

    passages = []
    last_first = max(len(sentences) - window, 0)
    for i in range(last_first + 1):
        group = sentences[i:i + window]
        start, end = group[0][0], group[-1][1]
        passages.append(Passage(id=len(passages) + 1, text=source_text[start:end], start=start, end=end))
    return passages


class EvidenceRetriever:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)  # slow: do this once
        self.passages: list[Passage] = []
        self._embeddings: torch.Tensor | None = None

    def index(self, source_text: str) -> int:
        """Prepare a source document for searching. Returns the passage count."""
        self.passages = build_passages(source_text)
        if self.passages:
            self._embeddings = self.model.encode(
                [p.text for p in self.passages],
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        else:
            self._embeddings = None
        return len(self.passages)

    def search(self, claim_text: str, top_k: int = 3) -> list[Evidence]:
        """Return the top_k passages most similar to the claim."""
        if self._embeddings is None:
            return []
        query = self.model.encode(
            claim_text, convert_to_tensor=True, normalize_embeddings=True
        )
        # Embeddings are normalized, so a dot product equals cosine similarity.
        scores = self._embeddings @ query
        k = min(top_k, len(self.passages))
        top = torch.topk(scores, k=k)
        return [
            Evidence(passage=self.passages[idx], score=float(score))
            for score, idx in zip(top.values.tolist(), top.indices.tolist())
        ]


if __name__ == "__main__":
    source = (
        "The trial enrolled 1,200 patients across six hospitals in 2022. "
        "Patients receiving the treatment recovered 12% faster than the control group. "
        "The treatment was well tolerated, with mild nausea as the most common side effect. "
        "The study was funded by a national health grant. "
        "Results were published in 2023."
    )
    claims = [
        "The treatment improved recovery rates by 47%.",
        "The study included 1,200 patients.",
        "The treatment was developed in Japan.",
    ]

    retriever = EvidenceRetriever()
    print("Passages indexed:", retriever.index(source))
    for claim in claims:
        print(f"\nCLAIM: {claim}")
        for ev in retriever.search(claim, top_k=2):
            print(f"  {ev.score:.3f}  {ev.passage.text}")