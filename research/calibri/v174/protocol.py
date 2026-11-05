"""Precommitted 2^3 source-compression intervention; diagnostics, not a codec."""
import re

SOURCE_BYTES = 11229303
SOURCE_SHA = "50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c"
TOTAL_UNIQUE = 134515008
FAMILIES = ("embedding_head", "attention", "mlp")
# All subsets, including both endpoints. No outcome-dependent arm selection.
MASKS = (0, 1, 3, 2, 6, 7, 5, 4)
MAX_NEW_TOKENS = 64
EXPECTED_CASES = 96


def family(path):
    if path in ("model.embed_tokens.weight", "lm_head.weight"):
        return "embedding_head"
    if re.fullmatch(r"model.layers.\d+.self_attn.(q|k|v|o)_proj.weight", path):
        return "attention"
    if re.fullmatch(r"model.layers.\d+.mlp.(gate|up|down)_proj.weight", path):
        return "mlp"
    raise ValueError("Unclassified trained matrix: " + path)


def selected(mask, path):
    if mask not in MASKS:
        raise ValueError("Unknown intervention mask")
    return bool(mask & (1 << FAMILIES.index(family(path))))


def arm_name(mask):
    if mask == 0:
        return "FP32"
    if mask == 7:
        return "ALL_CB59_DENSE_DIAGNOSTIC"
    return "CB59_" + "_and_".join(f for i, f in enumerate(FAMILIES) if mask & (1 << i))


def freeze_cases(cases):
    direct = [dict(c) for c in cases if c["style"] == "direct"]
    if len(direct) != EXPECTED_CASES or {c["question_id"] for c in direct} != set(range(96)):
        raise ValueError("Incomplete fixed behavioral suite")
    if len({c["prompt_sha"] for c in direct}) != EXPECTED_CASES:
        raise ValueError("Duplicate prompts")
    return direct
