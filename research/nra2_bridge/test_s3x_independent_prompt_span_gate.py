"""Synthetic negative controls for the research-only S3X gate.

These tests DO NOT attest S3T provenance, native tokenizer offsets, or 616 J7.
No model, gold, or protected data is loaded.
"""
import hashlib
import copy
import pytest

from research.nra2_bridge.s3x_independent_prompt_span_gate import verify_prompt_spans


def fixture():
    prompts = [f"case-{i:02d}: alpha then beta" for i in range(64)]
    hashes = [hashlib.sha256(p.encode("utf-8")).hexdigest() for p in prompts]
    mentions = []
    offsets = []
    for p in prompts:
        a = p.index("alpha")
        b = p.index("beta")
        mentions.append([
            {"start": a, "end": a + 5, "text": "alpha", "token_positions": [0]},
            {"start": b, "end": b + 4, "text": "beta", "token_positions": [1]},
        ])
        offsets.append([(a, a + 5), (b, b + 4)])
    return prompts, hashes, mentions, offsets


def test_synthetic_positive():
    assert verify_prompt_spans(*fixture()) is True


def test_prompt_tamper_rejected():
    p, h, m, o = fixture()
    p[0] += "!"
    with pytest.raises(AssertionError, match="Prompt SHA drift"):
        verify_prompt_spans(p, h, m, o)


def test_shifted_character_span_rejected():
    p, h, m, o = fixture()
    m[0][0]["start"] += 1
    with pytest.raises(AssertionError):
        verify_prompt_spans(p, h, m, o)


def test_shifted_token_offset_rejected():
    p, h, m, o = fixture()
    o[0][0] = (o[0][0][0] + 1, o[0][0][1])
    with pytest.raises(AssertionError, match="Token offset ABI drift"):
        verify_prompt_spans(p, h, m, o)


def test_unaccounted_overlap_rejected():
    p, h, m, o = fixture()
    a = m[0][0]["start"]
    o[0].append((a, a + 2))
    with pytest.raises(AssertionError, match="Unaccounted overlapping token"):
        verify_prompt_spans(p, h, m, o)


def test_missing_record_rejected():
    p, h, m, o = fixture()
    with pytest.raises(AssertionError, match="exactly 64"):
        verify_prompt_spans(p[:-1], h, m, o)


def test_forged_but_self_consistent_offsets_are_not_attested():
    # Deliberately documents the trust boundary: offset arrays supplied by a
    # caller are NOT proof of actual native-tokenizer execution.
    p, h, m, o = fixture()
    assert verify_prompt_spans(p, h, m, o) is True
    # No independent tokenizer capture is performed here; physical S3U remains
    # BLOCKED regardless of this passing synthetic example.


def test_independent_retokenization_accepts_matching_offsets():
    from research.nra2_bridge.s3x_independent_prompt_span_gate import verify_with_retokenization
    p, h, m, o = fixture()
    def tokenizer(text):
        a, b = text.index("alpha"), text.index("beta")
        return [(a, a + 5), (b, b + 4)]
    assert verify_with_retokenization(p, h, m, o, tokenizer) is True


def test_independent_retokenization_rejects_forged_self_consistent_offsets():
    from research.nra2_bridge.s3x_independent_prompt_span_gate import verify_with_retokenization
    p, h, m, o = fixture()
    # Caller changes both mention and offset to a wrong but matching substring.
    m[0][0]["start"] += 1
    m[0][0]["end"] += 1
    m[0][0]["text"] = p[0][m[0][0]["start"]:m[0][0]["end"]]
    o[0][0] = (m[0][0]["start"], m[0][0]["end"])
    assert verify_prompt_spans(p, h, m, o) is True
    def tokenizer(text):
        a, b = text.index("alpha"), text.index("beta")
        return [(a, a + 5), (b, b + 4)]
    with pytest.raises(AssertionError, match="Independent tokenizer offset mismatch"):
        verify_with_retokenization(p, h, m, o, tokenizer)


def test_physical_prereg_rejects_recomputed_candidate_digest():
    from research.nra2_bridge.s3x_independent_prompt_span_gate import verify_s3t_prereg_bytes
    candidate = b"synthetic-not-the-physical-S3T-artifact"
    with pytest.raises(AssertionError, match="SHA mismatch"):
        verify_s3t_prereg_bytes(candidate)
    # A self-derived SHA is not an acceptable replacement for the frozen pin.
    with pytest.raises(AssertionError, match="Pinned S3T"):
        verify_s3t_prereg_bytes(candidate, hashlib.sha256(candidate).hexdigest())


def test_physical_prereg_rejects_nonbytes():
    from research.nra2_bridge.s3x_independent_prompt_span_gate import verify_s3t_prereg_bytes
    with pytest.raises(AssertionError, match="Pinned S3T"):
        verify_s3t_prereg_bytes("synthetic")
