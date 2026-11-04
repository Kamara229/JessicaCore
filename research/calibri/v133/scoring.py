"""Scoring stays observational. Do NOT call lexical proxies semantic accuracy."""
from v130.scoring import evidence, summarize
from .arms import ARMS


def evaluate_case(tokenizer, pair, side, question, gold, foil, generation):
    tokens = list(generation["ids"])
    eos_id = tokenizer.eos_token_id
    if any(not isinstance(x, int) for x in tokens):
        raise ValueError("nonnumeric token")
    obs = evidence(generation["text"], question, gold, foil, generation["eos_generated"])
    return {
        "pair": pair, "side": side, "question": question,
        "reference_for_post_generation_grading_only": gold,
        "foil_for_post_generation_grading_only": foil,
        "generated_text": generation["text"],
        "generated_token_ids": tokens,
        "first_generated_token_id": tokens[0] if tokens else None,
        "first_generated_token_is_EOS": bool(tokens and tokens[0] == eos_id),
        "observation": obs,
    }


def report(arms):
    if set(arms) != set(ARMS):
        raise ValueError("missing arm")
    anchors = [(r["pair"],r["side"],r["question"]) for r in arms["FP32"]]
    if len(anchors) != 16 or len(set(anchors)) != 16:
        raise ValueError("paired corpus mismatch")
    summaries = {}
    for key,rows in arms.items():
        if [(r["pair"],r["side"],r["question"]) for r in rows] != anchors:
            raise ValueError("comparison not prompt matched")
        summaries[key] = {
            **summarize(rows),
            "first_token_is_EOS": sum(r["first_generated_token_is_EOS"] for r in rows),
            "mean_generated_tokens": sum(len(r["generated_token_ids"]) for r in rows) / len(rows),
            "mean_repetition_distinct_words_ratio": sum(r["observation"]["distinct_word_ratio"] for r in rows) / len(rows),
        }
    neutral = arms["v119_neutral"]
    ablated = arms["v119_drop_band0"]
    green = arms["v119_drop_band0_green_current"]
    swapped = arms["v119_drop_band0_green_swapped"]
    comparisons = {
        "drop_band0_changes_generated_tokens": sum(a["generated_token_ids"] != b["generated_token_ids"]
                                                  for a,b in zip(neutral,ablated)),
        "drop_band0_eliminates_first_EOS": sum(
            a["first_generated_token_is_EOS"] and not b["first_generated_token_is_EOS"]
            for a,b in zip(neutral,ablated)),
        "green_current_vs_swapped_changes_generated_tokens": sum(
            a["generated_token_ids"] != b["generated_token_ids"]
            for a,b in zip(green,swapped)),
        "gold_unambiguous_both_sides_by_arm": {
            arm: summaries[arm]["both_sides_unique_target_visible"] for arm in ARMS},
    }
    return {"automated_visibility_ONLY": summaries,
            "paired_controls": comparisons,
            "blind_human_semantic_review_required": True,
            "claim": "Dropping first-token EOS is not evidence of sensible answers."}
