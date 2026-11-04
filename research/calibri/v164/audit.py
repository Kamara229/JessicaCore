"""Source TRAIN FIT 16-arm causal operator NLL and teacher KL deltas.

Single-operator and leave-one-out effects are observational under the
old data: per-operator contribution is NOT independent, 7 operators
interact nonlinearly. No causal guarantee on unseen corpora or QA.
"""
from v142.scoring import aggregate
from .protocol import (ARMS,FIT_DOMAINS,SHORT,WINDOWS,WINDOW,MATRIX_KEYS)


def summarize(rows,timings):
    if tuple(rows)!=ARMS or tuple(timings)!=ARMS:
        raise ValueError("all 16 predeclared source operator arms required")
    metrics={}
    for arm in ARMS:
        if tuple(rows[arm])!=FIT_DOMAINS:
            raise ValueError("training source corpus missing")
        metrics[arm]={}
        for domain in FIT_DOMAINS:
            data=rows[arm][domain]
            if len(data)!=WINDOWS or any(
                    int(x["tokens_scored"])!=WINDOW-1 for x in data):
                raise ValueError("original four matched TRAIN FIT windows not complete")
            metrics[arm][domain]=aggregate(data)
    overall={}
    for domain in FIT_DOMAINS:
        ref=metrics[ARMS[0]][domain]
        entire=metrics[ARMS[1]][domain]
        def delta(arm):
            src=metrics[arm][domain]
            return {
                "delta_TRUE_next_token_NLL_vs_CB59":
                    src["mean_TRUE_next_token_NLL"]-ref["mean_TRUE_next_token_NLL"],
                "delta_teacher_KL_vs_CB59":
                    src["teacher_to_candidate_KL"]-ref["teacher_to_candidate_KL"],
                "teacher_argmax_agreement_fraction":
                    src["teacher_next_token_argmax_agreement_fraction"],
            }
        records=[]
        for key,short in zip(MATRIX_KEYS,SHORT):
            one=delta("paid_ONLY_"+short)
            except_arm=metrics["paid_EXCEPT_"+short][domain]
            records.append({
                "original_trained_operator":key,
                "only_operator_vs_original":one,
                "removing_operator_from_ALL7_change_NLL":
                    except_arm["mean_TRUE_next_token_NLL"]-
                    entire["mean_TRUE_next_token_NLL"],
                "removing_operator_from_ALL7_change_KL":
                    except_arm["teacher_to_candidate_KL"]-
                    entire["teacher_to_candidate_KL"],
                "cannot_extrapolate_tiny_train_fit_to_new_VALID":True,
            })
        overall[domain]={
            "all7_vs_original_CB59":delta(ARMS[1]),
            "diagnostic_7_individual_and_7_leave_one_out":records,
            "best_single_by_OLD_TRAIN_FIT_NLL_only_NOT_VALID":
                min(records,key=lambda x:x["only_operator_vs_original"][
                    "delta_TRUE_next_token_NLL_vs_CB59"])["original_trained_operator"],
            "whole_model_native_cpu_seconds":
                {arm:timings[arm][domain] for arm in ARMS},
        }
    return {
        "per_previous_original_TRAIN_FIT_source":overall,
        "full_16arm_NLL_KL_metrics":metrics,
        "this_is_mechanistic_hypothesis_NOT_quality_validation":True,
        "NO_additional_physical_parameters_credited_or_added":True,
        "NO_untouched_v89_TEST_or_new_QA_access":True,
    }
