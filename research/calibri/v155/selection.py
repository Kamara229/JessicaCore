"""Important: original v154 best and worst are NOT source VALID winners."""
from v154.arms import subset
from v155.protocol import ARM_NAMES,DOMAINS

def frozen_arms(physical,decision):
    best=decision["best"]
    worst=decision["worst"]
    if best==worst:raise ValueError("cannot assess best-vs-worst operator")
    return {
        ARM_NAMES[0]:subset(physical,()),
        ARM_NAMES[1]:subset(physical,tuple(physical.patches)),
        ARM_NAMES[2]:subset(physical,(best,)),
        ARM_NAMES[3]:subset(physical,(worst,)),
    }

def holdout_gate(aggregated):
    if tuple(aggregated)!=ARM_NAMES:raise ValueError("all four arms required")
    out={}
    for domain in DOMAINS:
        base=aggregated[ARM_NAMES[0]][domain]
        control=aggregated[ARM_NAMES[1]][domain]
        chosen=aggregated[ARM_NAMES[2]][domain]
        worst=aggregated[ARM_NAMES[3]][domain]
        out[domain]={
            "single_best_minus_CB59_NLL":chosen["mean_TRUE_next_token_NLL"]-base["mean_TRUE_next_token_NLL"],
            "single_best_minus_CB59_teacher_KL":chosen["teacher_to_candidate_KL"]-base["teacher_to_candidate_KL"],
            "single_best_minus_all7_NLL":chosen["mean_TRUE_next_token_NLL"]-control["mean_TRUE_next_token_NLL"],
            "single_worst_minus_CB59_NLL":worst["mean_TRUE_next_token_NLL"]-base["mean_TRUE_next_token_NLL"],
            "best_teacher_top1_match":chosen["teacher_next_token_argmax_agreement_fraction"],
        }
    passed=all(v["single_best_minus_CB59_NLL"]<0 and
               v["single_best_minus_CB59_teacher_KL"]<=0 for v in out.values())
    return out,passed
