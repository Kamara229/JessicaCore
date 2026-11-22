"""Independent, read-only S3T prompt/mention verification; no labels or weights.

Inputs MUST originate from separately pinned physical S3T bytes and native
tokenizer offset mappings. This helper cannot establish their provenance.
"""
import hashlib


def verify_prompt_spans(prompts, expected_sha256, mentions, token_offsets):
    """Fail closed; return True only for 64 byte-matched, span-aligned prompts.

    mentions[i]: two dicts {start,end,text,token_positions}, in text order.
    token_offsets[i]: native tokenizer (start,end) codepoint offsets.
    Reject special tokens (0,0) as mention tokens and partial-token spans.
    """
    if not (len(prompts) == len(expected_sha256) == len(mentions)
            == len(token_offsets) == 64):
        raise AssertionError("Expected exactly 64 ordered records")
    if len(set(expected_sha256)) != 64:
        raise AssertionError("Duplicate prompt hashes")
    for i, (prompt, digest, pair, offsets) in enumerate(
            zip(prompts, expected_sha256, mentions, token_offsets)):
        if not isinstance(prompt, str) or not isinstance(digest, str):
            raise AssertionError(f"Invalid prompt record {i}")
        actual = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if actual != digest:
            raise AssertionError(f"Prompt SHA drift at {i}")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise AssertionError(f"Expected two mentions at {i}")
        previous_end = 0
        occupied = set()
        for mention in pair:
            start, end = mention["start"], mention["end"]
            if (type(start) is not int or type(end) is not int
                    or not previous_end <= start < end <= len(prompt)):
                raise AssertionError(f"Character span drift at {i}")
            if prompt[start:end] != mention["text"]:
                raise AssertionError(f"Mention text drift at {i}")
            positions = mention["token_positions"]
            if (not isinstance(positions, (list, tuple)) or not positions
                    or any(type(p) is not int or p < 0 or p >= len(offsets)
                           for p in positions)
                    or list(positions) != list(range(positions[0], positions[-1] + 1))
                    or occupied.intersection(positions)):
                raise AssertionError(f"Token positions drift at {i}")
            occupied.update(positions)
            spans = [offsets[p] for p in positions]
            if (any(type(a) is not int or type(b) is not int
                    or a >= b for a, b in spans)
                    or spans[0][0] != start or spans[-1][1] != end
                    or any(spans[k][1] != spans[k+1][0]
                           for k in range(len(spans)-1))):
                raise AssertionError(f"Token offset ABI drift at {i}")
            # Reject an unaccounted overlapping native token.
            for p, (a, b) in enumerate(offsets):
                if p not in occupied and a < end and b > start and a < b:
                    raise AssertionError(f"Unaccounted overlapping token at {i}")
            previous_end = end
    return True


def verify_with_retokenization(prompts, expected_sha256, mentions,
                               supplied_offsets, tokenizer_offsets_fn):
    """Reject caller-forged offset arrays by independently retokenizing all 64.

    tokenizer_offsets_fn must be a separately pinned native S194 tokenizer
    implementation returning complete offset_mapping for one prompt. This
    function cannot attest that the callable really is that tokenizer.
    """
    if not callable(tokenizer_offsets_fn):
        raise AssertionError("Independent tokenizer callable required")
    if len(prompts) != 64 or len(supplied_offsets) != 64:
        raise AssertionError("Expected exactly 64 ordered records")
    recomputed = []
    for i, prompt in enumerate(prompts):
        try:
            actual = tokenizer_offsets_fn(prompt)
            actual = [tuple(pair) for pair in actual]
        except Exception as exc:
            raise AssertionError(f"Tokenizer offset capture failed at {i}") from exc
        if actual != [tuple(pair) for pair in supplied_offsets[i]]:
            raise AssertionError(f"Independent tokenizer offset mismatch at {i}")
        recomputed.append(actual)
    return verify_prompt_spans(prompts, expected_sha256, mentions, recomputed)


# Physical source prerequisite; the expected digest is from the frozen S3T
# preregistration, never derived from the candidate bytes being checked.
S3T_PREREG_SHA256 = "034ebd6e1cf6f20ad04ccd00e70c6bc07ee08a46f47da843121524f1d7f0faf3"


def verify_s3t_prereg_bytes(source_bytes, expected_sha256=S3T_PREREG_SHA256):
    """Fail closed on missing/drifted physical S3T preregistration bytes.

    The caller must supply the EXACT byte sequence whose SHA was preregistered;
    this function does not assume ZIP/JSON format, extract prompts, or attest
    that the bytes came from GitHub Actions. Never substitute a reserialized
    JSON or a derived prompt manifest for the pinned physical artifact.
    """
    if type(source_bytes) is not bytes or expected_sha256 != S3T_PREREG_SHA256:
        raise AssertionError("Pinned S3T preregistration bytes and SHA required")
    actual = hashlib.sha256(source_bytes).hexdigest()
    if actual != S3T_PREREG_SHA256:
        raise AssertionError("Physical S3T preregistration SHA mismatch")
    return True
