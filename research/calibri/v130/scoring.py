"""Conservative observable text measures, not a substitute for blind fact grading."""
import re
from v128.scoring import matches


def evidence(text, question, gold, foil, eos):
    # Spot both alternatives in the actual answer, not in the prompt.
    # A response repeating both facts must not pass as a correct answer.
    tokens = set(re.findall(r"[a-z0-9]+", text.casefold()))
    target = gold.casefold() in tokens
    contradiction = foil.casefold() in tokens
    exact = matches(text, gold, question)
    distinct = len(set(re.findall(r"[a-z0-9]+", text.casefold())))
    all_words = len(re.findall(r"[a-z0-9]+", text.casefold()))
    return {
        "strict_exact_text": bool(exact),
        "strict_exact_with_eos": bool(exact and eos),
        "target_visible_alone_NOT_proven_semantic": bool(target and not contradiction),
        "both_alternatives_visible": bool(target and contradiction),
        "foil_visible_alone": bool(contradiction and not target),
        "no_alternative_visible": bool(not target and not contradiction),
        "ended_by_eos": bool(eos),
        "distinct_word_ratio": (distinct / all_words if all_words else 0.0),
    }


def summarize(rows):
    pair_ids = sorted({r["pair"] for r in rows})
    by_pair = {p: {r["side"]: r for r in rows if r["pair"] == p} for p in pair_ids}
    assert all(set(b) == {"left", "right"} for b in by_pair.values())
    fields = ("strict_exact_text", "strict_exact_with_eos",
              "target_visible_alone_NOT_proven_semantic", "both_alternatives_visible",
              "foil_visible_alone", "no_alternative_visible", "ended_by_eos")
    report = {key: sum(bool(r["observation"][key]) for r in rows) for key in fields}
    report["number_of_questions"] = len(rows)
    report["both_sides_strict_exact"] = sum(all(r["observation"]["strict_exact_text"]
         for r in v.values()) for v in by_pair.values())
    report["both_sides_unique_target_visible"] = sum(all(
         r["observation"]["target_visible_alone_NOT_proven_semantic"]
         for r in v.values()) for v in by_pair.values())
    report["number_of_counterfactual_pairs"] = len(pair_ids)
    report["blind_semantic_review_needed"] = True
    return report
