"""Summarize effect of physically existing body residuals on EOS and lexical rank."""
from statistics import mean

def summarize(arms):
    if "head_only" not in arms or "complete" not in arms:
        raise ValueError("missing anchors")
    ids = [(r["pair"], r["side"]) for r in arms["head_only"]]
    if len(ids) != 16 or len(set(ids)) != 16:
        raise ValueError("pair identities inconsistent")
    stats = {}
    for arm, rows in arms.items():
        if [(r["pair"], r["side"]) for r in rows] != ids:
            raise ValueError(f"not a matched pair comparison: {arm}")
        stats[arm] = {
            "first_eos_argmax": sum(r["eos_first"] for r in rows),
            "gold_first_token_top8_PROXY": sum(r["best_expected_rank"] <= 8 for r in rows),
            "mean_eos_logit_margin": mean(r["eos_margin"] for r in rows),
            "mean_eos_probability": mean(r["eos_probability"] for r in rows),
        }
    paired = {}
    for band in range(6):
        without = arms[f"drop_band_{band}"]
        alone = arms[f"only_band_{band}"]
        full = arms["complete"]
        head = arms["head_only"]
        paired[str(band)] = {
            "drop_band_reverses_full_EOS": sum(
                f["eos_first"] and not w["eos_first"] for f, w in zip(full, without)),
            "drop_band_creates_EOS": sum(
                not f["eos_first"] and w["eos_first"] for f, w in zip(full, without)),
            "only_band_creates_EOS_vs_head_only": sum(
                not h["eos_first"] and a["eos_first"] for h, a in zip(head, alone)),
            "mean_margin_change_when_dropped_from_complete": mean(
                w["eos_margin"] - f["eos_margin"] for f, w in zip(full, without)),
            "mean_margin_change_when_added_to_head_only": mean(
                a["eos_margin"] - h["eos_margin"] for h, a in zip(head, alone)),
            "number_of_cases": len(full),
        }
    return {
        "by_arm": stats, "by_band": paired,
        "test_scope": "FIRST TOKEN pilot on 8 fixed pairs: not complete-response accuracy",
        "interpretation_limit": "necessity/sufficiency probes are nonlinear and overlapping; do not assign uniquely additive blame",
    }
