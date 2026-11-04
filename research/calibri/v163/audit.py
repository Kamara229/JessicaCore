"""Report raw answer accuracy and teacher agreement; no false release gate."""
from .protocol import ARMS,PACKED_ARMS,DATASETS,COUNT,ROW_BEGIN,ROW_END,GENERATION_INDICES


def summarize_choices(per_arm):
    if tuple(per_arm)!=ARMS:
        raise ValueError("missing pretrained FP32 or original packed comparison")
    final={}
    for domain in DATASETS:
        seen=[(x["row"],x["prompt_sha"]) for x in per_arm[ARMS[0]][domain]]
        if len(seen)!=COUNT or [x[0] for x in seen]!=list(range(ROW_BEGIN,ROW_END)):
            raise ValueError("missing QA domain source rows")
        for arm in ARMS[1:]:
            if [(x["row"],x["prompt_sha"]) for x in per_arm[arm][domain]]!=seen:
                raise ValueError("different questions across original models")
        expected=per_arm[ARMS[0]][domain]
        summary={}
        for arm in ARMS:
            records=per_arm[arm][domain]
            correct=sum(x["correct"] for x in records)
            summary[arm]={
               "correct_out_of_16":correct,
               "accuracy":correct/COUNT,
               "FP32_teacher_forced_choice_agreement_count":sum(
                   x["chosen_index"]==teacher["chosen_index"]
                   for x,teacher in zip(records,expected)),
               "mean_gold_label_conditional_probability":sum(
                   x["gold_conditional_probability"] for x in records)/COUNT,
               "exact_all_16_source_prompts_scored":True,
            }
        summary["paid_block0_minus_original_CB59_correct"]=(
            summary[ARMS[2]]["correct_out_of_16"]-
            summary[ARMS[1]]["correct_out_of_16"])
        summary["no_QA_regression_from_paid_block0"]=(
            summary["paid_block0_minus_original_CB59_correct"]>=0)
        final[domain]=summary
    return {
       "fresh_science_boolean_question_accuracy":final,
       "no_2domain_forced_choice_regression":all(
           final[d]["no_QA_regression_from_paid_block0"] for d in DATASETS),
       "16_per_domain_too_small_for_conclusive_semantic_claim":True,
       "no_teacher_argmax_alignment_implied_from_true_NLL_KL":True,
    }


def summarize_generation(raw):
    if tuple(raw)!=PACKED_ARMS:
        raise ValueError("both real packed model greedy responses missing")
    summaries={}
    for arm in PACKED_ARMS:
        rows=raw[arm]
        if len(rows)!=len(GENERATION_INDICES) or [
            r["prior_historical_prompt_index"] for r in rows
        ]!=list(GENERATION_INDICES):
            raise ValueError("missing four exact historical prompts")
        ids=[r["actual_full_generated_token_ids"] for r in rows]
        summaries[arm]={
          "genuine_full_model_greedy_prompts":len(rows),
          "actual_generated_token_count":sum(len(x) for x in ids),
          "distinct_token_ratio_over_all_four_responses":
              sum(len(set(x))/max(len(x),1) for x in ids)/len(ids),
          "actual_eos_generated_count":sum(r["EOS_generated"] for r in rows),
          "full_model_seconds_total":sum(r["full_model_wall_seconds"] for r in rows),
          "sampled_actual_text_and_token_ids":rows,
        }
    seconds={a:summaries[a]["full_model_seconds_total"] for a in PACKED_ARMS}
    generated={a:summaries[a]["actual_generated_token_count"] for a in PACKED_ARMS}
    if any(t<=0 or seconds[a]<=0 for a,t in generated.items()):
        raise ValueError("invalid full token generation measurement")
    baseline_speed=generated[PACKED_ARMS[0]]/seconds[PACKED_ARMS[0]]
    paid_speed=generated[PACKED_ARMS[1]]/seconds[PACKED_ARMS[1]]
    return {
       "genuine_full_model_autoregressive_measures":summaries,
       "genuine_packed_tokens_per_second_ratio_paid_vs_CB59":
          paid_speed/baseline_speed,
       "not_a_statistically_stable_cpu_speed_release_benchmark":True,
       "no_automatic_coherent_or_correct_generation_claim":True,
    }
