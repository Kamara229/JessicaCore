"""Choose gains ONLY from the first 2 original TRAIN-VALID windows/domain.

The second 2 windows/domain are never accessed by this function.
Selection aims to avoid cross-domain regression on BOTH NLL and teacher KL.
If no grid point passes strict preregistered gates, fallback to CB59.
"""
from v142.scoring import aggregate
from .protocol import DOMAINS,GAIN_GRID,BASELINE,label

NLL_MIN_GAIN_PER_DOMAIN=.05
KL_MAX_REGRESSION=0.0


def choose(selection):
    expected={label(*pair) for pair in GAIN_GRID}
    if set(selection)!=expected:
        raise ValueError("incomplete 25-gain preregistered source selection")
    summaries={arm:{d:aggregate(records[d]) for d in DOMAINS}
               for arm,records in selection.items()}
    base=summaries[BASELINE]
    candidates=[]
    for i,h in GAIN_GRID:
        arm=label(i,h)
        if arm==BASELINE:continue
        change={d:{
            "NLL_gain":base[d]["mean_TRUE_next_token_NLL"]-
                       summaries[arm][d]["mean_TRUE_next_token_NLL"],
            "KL_gain":base[d]["teacher_to_candidate_KL"]-
                      summaries[arm][d]["teacher_to_candidate_KL"],
        } for d in DOMAINS}
        if all(change[d]["NLL_gain"]>=NLL_MIN_GAIN_PER_DOMAIN and
               change[d]["KL_gain"]>=-KL_MAX_REGRESSION for d in DOMAINS):
            candidates.append((min(change[d]["NLL_gain"] for d in DOMAINS),
                               min(change[d]["KL_gain"] for d in DOMAINS),
                               arm))
    if not candidates:
        return BASELINE,{"no_robust_both_domain_candidate":True,
                         "chosen_from_FIRST_TWO_windows_only":True,
                         "select_rule":"Both domain NLL >=0.05 better and KL no worse vs CB59",
                         "selection_summary":summaries}
    best=max(candidates)
    return best[2],{"no_robust_both_domain_candidate":False,
                    "worst_domain_NLL_gain":best[0],
                    "worst_domain_KL_gain":best[1],
                    "chosen_from_FIRST_TWO_windows_only":True,
                    "select_rule":"Both domain NLL >=0.05 better and KL no worse vs CB59",
                    "selection_summary":summaries}
