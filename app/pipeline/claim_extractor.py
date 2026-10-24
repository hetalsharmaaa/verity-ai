"""Claim extraction: split AI-generated text into individual checkable claims.

Version 1 is rule-based sentence splitting. Each claim keeps its character
offsets in the original text so the UI can highlight it later.
"""

import re
from dataclasses import dataclass

# A sentence boundary is: end punctuation, whitespace, then a capital letter,
# digit, or opening quote/bracket. Decimals like "47.5%" have no whitespace
# after the dot, so they are never split. A newline also ends a segment.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|\n+")

# Leading list markers such as "- ", "* ", "1. " or "2) ".
_LIST_MARKER = re.compile(r"^(?:[-*\u2022]|\d+[.)])\s+")

# Lowercase tokens that end with a period but do NOT end a sentence.
_ABBREVIATIONS = {
    "dr.", "mr.", "mrs.", "ms.", "prof.", "vs.", "etc.", "e.g.", "i.e.",
    "fig.", "inc.", "ltd.", "approx.", "al.",
}

MIN_WORDS = 4


@dataclass
class Claim:
    id: int
    text: str
    start: int  # character offset in the original text (inclusive)
    end: int    # character offset in the original text (exclusive)


def _ends_with_abbreviation(candidate: str) -> bool:
    tokens = candidate.strip().split()
    return bool(tokens) and tokens[-1].lower() in _ABBREVIATIONS


def _split_segments(text: str) -> list[tuple[int, int]]:
    """Return (start, end) offsets of raw sentence segments."""
    segments = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        if _ends_with_abbreviation(text[start:match.start()]):
            continue  # e.g. "Dr. Smith": not a real boundary
        segments.append((start, match.start()))
        start = match.end()
    segments.append((start, len(text)))
    return segments


def _is_checkable(sentence: str) -> bool:
    """Skip questions and fragments that cannot be verified."""
    if sentence.endswith("?"):
        return False
    return len(sentence.split()) >= MIN_WORDS


def extract_claims(text: str) -> list[Claim]:
    """Split text into checkable claims with offsets into the original text."""
    claims = []
    for seg_start, seg_end in _split_segments(text):
        raw = text[seg_start:seg_end]
        stripped = raw.strip()
        if not stripped:
            continue

        start = seg_start + (len(raw) - len(raw.lstrip()))

        marker = _LIST_MARKER.match(stripped)
        if marker:
            start += marker.end()
            stripped = stripped[marker.end():]

        if not _is_checkable(stripped):
            continue

        claims.append(
            Claim(id=len(claims) + 1, text=stripped, start=start, end=start + len(stripped))
        )
    return claims


if __name__ == "__main__":
    sample = (
        "A 2023 study found that this treatment improved recovery rates by 47.5%. "
        "Dr. Smith led the research at Oxford. Is this reliable? "
        "The trial included 1,200 patients."
    )
    for claim in extract_claims(sample):
        print(claim)