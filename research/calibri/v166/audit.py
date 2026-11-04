"""Report causal teacher-state transplantation effect, not compact success."""
from v142.scoring import aggregate
from .protocol import FIT_DOMAINS,ARM_NAMES,TEACHER_ONLY_STAGES,WINDOWS,WINDOW


def summarize(per_arm):
    if tuple(per_arm)!=ARM_NAMES:
        raise ValueError("missing original native CB59 and fully paid original CB161")
    metrics={}
    for arm in ARM_NAMES:
        records=per_arm[arm]
        if tuple(records)!=FIT_DOMAINS:
            raise ValueError("missing actual original FIT corpus")
        metrics[arm]={}
        for domain in FIT_DOMAINS:
            parts=records[domain]
            expected=("baseline",)+TEACHER_ONLY_STAGES
            if tuple(parts)!=expected:
                raise ValueError("missing one predeclared teacher oracle intervention")
            by_stage={}
            for stage in expected:
                rows=parts[stage]
                if len(rows)!=WINDOWS or any(x["tokens_scored"]!=WINDOW-1 for x in rows):
                    raise ValueError("missing source 4x64 window logits for actual model")
                by_stage[stage]=aggregate(rows)
            metrics[arm][domain]=by_stage
    analysis={}
    for arm in ARM_NAMES:
        analysis[arm]={}
        for domain in FIT_DOMAINS:
            source=metrics[arm][domain]
            baseline=source["baseline"]
            result={}
            for stage in TEACHER_ONLY_STAGES:
                alt=source[stage]
                result[stage]={
                    "teacher_oracle_injected_source_stage":stage,
                    "UNPAID_activation_ORACLE_NLL_delta_vs_original_packed":
                        alt["mean_TRUE_next_token_NLL"]-baseline["mean_TRUE_next_token_NLL"],
                    "UNPAID_activation_ORACLE_FP32_KL_delta_vs_original_packed":
                        alt["teacher_to_candidate_KL"]-baseline["teacher_to_candidate_KL"],
                    "UNPAID_activation_ORACLE_teacher_top1_agreement":
                        alt["teacher_next_token_argmax_agreement_fraction"],
                    "pure_teacher_hidden_restoration_NOT_deployable":True,
                }
            best_by_kl=min(result,key=lambda key:result[key][
                "UNPAID_activation_ORACLE_FP32_KL_delta_vs_original_packed"])
            analysis[arm][domain]={
                "exact_original_baseline_metrics":baseline,
                "five_oracle_hidden_state_boundary_effects":result,
                "LOWEST_FP32_KL_unpaid_teacher_oracle_stage_on_old_FIT":best_by_kl,
                "cannot_rank_trainable_GT_generalization_from_old_TRAIN_FIT":True,
            }
    return {
        "actual_old_TRAIN_FIT_source_only":analysis,
        "full_next_token_NLL_KL_PPL_teacher_top1_by_stage":metrics,
        "theoretical_teacher_activation_transplant_NOT_a_trained_10MB_Green_Tunnel":True,
        "even_ORACLE_improvement_does_NOT_prove_student_can_infer_reference_state":True,
        "no_original_compressed_archival_0p7bpp_success_claim_from_an_oracle":True,
    }
