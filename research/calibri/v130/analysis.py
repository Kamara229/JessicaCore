"""Post-hoc causal controls on already generated, immutable trajectories.

Never promote correctness from rank, token presence, or EOS alone. Conditioning
on FP32 competence is reported alongside all-pair results to avoid conflating
a compressed-host failure with a base-model task failure.
"""
from collections import defaultdict


ARMS = ("FP32", "CB59", "v119_neutral", "v119_green_current",
        "v119_green_swapped", "v119_green_constant")
OBSERVATIONS = ("strict_exact_text", "strict_exact_with_eos",
                "target_visible_alone_NOT_proven_semantic")


def by_pair(rows):
    grouped = defaultdict(dict)
    for row in rows:
        pair, side = row["pair"], row["side"]
        if side not in ("left", "right") or side in grouped[pair]:
            raise ValueError("duplicate or unknown pair side")
        grouped[pair][side] = row
    if not grouped or any(set(group) != {"left", "right"} for group in grouped.values()):
        raise ValueError("missing counterfactual side")
    return dict(grouped)


def paired_score(group, field):
    return all(group[side]["observation"][field] for side in ("left", "right"))


def compare(arms):
    if set(arms) != set(ARMS):
        raise ValueError("missing, duplicate, or unexpected comparison arm")
    groups = {name: by_pair(arms[name]["rows"]) for name in ARMS}
    names = sorted(groups["FP32"])
    if any(sorted(groups[k]) != names for k in ARMS):
        raise ValueError("not identical counterfactual cases")
    # Verify same prompts and gold labels were used for every arm.
    for pair in names:
        for side in ("left", "right"):
            original = groups["FP32"][pair][side]
            for name in ARMS[1:]:
                row = groups[name][pair][side]
                for key in ("question", "reference_for_post_generation_grading_only"):
                    if row[key] != original[key]:
                        raise ValueError(f"unmatched {pair}/{side}/{name}/{key}")

    view = {
        "scope": "pilot only; 8 constructed pairs; no statistical generalization",
        "FP32_is_reference_not_ORACLE_during_generation": True,
        "FP32_failure_does_not_count_as_proven_quantization_information_loss": True,
        "Green_swap_is_counterfactual_control_not_independent_randomization": True,
        "complete_paired_rows": len(names),
        "metrics": {},
    }
    for metric in OBSERVATIONS:
        eligible = [pair for pair in names if paired_score(groups["FP32"][pair], metric)]
        metrics = {
            "fp32_pair_eligible": len(eligible),
            "fp32_eligible_pair_names": eligible,
            "FP32_unproven_pairs": len(names) - len(eligible),
            "all_pairs": {},
            "eligible_pairs": {},
        }
        for arm in ARMS:
            metrics["all_pairs"][arm] = sum(paired_score(groups[arm][p], metric)
                                           for p in names)
            metrics["eligible_pairs"][arm] = sum(paired_score(groups[arm][p], metric)
                                                for p in eligible)
        # A gain over all controls is a NECESSARY pilot signal, not sufficient
        # evidence for inference or for deployment.
        metrics["green_current_unique_wins_vs_both_controls"] = sum(
            paired_score(groups["v119_green_current"][p], metric) and
            not paired_score(groups["v119_green_swapped"][p], metric) and
            not paired_score(groups["v119_green_constant"][p], metric)
            for p in eligible)
        view["metrics"][metric] = metrics

    # Quantify responses that react to changed bindings. This is not accuracy.
    view["generation_changed_between_counterfactuals"] = {
        arm: sum(groups[arm][p]["left"]["generated_token_ids"] !=
                 groups[arm][p]["right"]["generated_token_ids"] for p in names)
        for arm in ARMS}
    view["controller_current_vs_swapped_changes_tokens"] = sum(
        groups["v119_green_current"][p][side]["generated_token_ids"] !=
        groups["v119_green_swapped"][p][side]["generated_token_ids"]
        for p in names for side in ("left", "right"))
    view["conclusion"] = "DIAGNOSTIC_ONLY; no GREEN causal efficacy claim."
    return view
