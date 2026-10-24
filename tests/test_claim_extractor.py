from app.pipeline.claim_extractor import extract_claims


def test_splits_two_sentences():
    claims = extract_claims("Paris is the capital of France. Berlin is the capital of Germany.")
    assert len(claims) == 2


def test_does_not_split_decimals():
    claims = extract_claims("The drug improved recovery by 47.5% in 2023. It was well tolerated.")
    assert len(claims) == 2
    assert "47.5%" in claims[0].text


def test_does_not_split_after_abbreviation():
    claims = extract_claims("Dr. Smith led the study at Oxford. It ran for two years.")
    assert len(claims) == 2
    assert claims[0].text.startswith("Dr. Smith")


def test_skips_questions():
    claims = extract_claims("Is this really true? The sky is blue today.")
    assert len(claims) == 1


def test_offsets_match_original_text():
    text = "- Water boils at 100 degrees Celsius.\n- The moon orbits the Earth."
    for claim in extract_claims(text):
        assert text[claim.start:claim.end] == claim.text