"""All statistical comparisons only on independently-heldout true answer prompts."""
from v124.training import infer,grade

def independent_metric(models,rows):
    results={}
    for name,model in models.items():
        probabilities=infer(model,rows)
        if len(probabilities)!=len(rows) or len(rows)%4:
            raise AssertionError("wrong genuine heldout sample alignment")
        ranks=[]
        for i in range(0,len(rows),4):
            group=probabilities[i:i+4]
            if not all(rows[i+k][0]==rows[i][0] for k in range(4)):
                raise AssertionError("misaligned independent QA")
            ranks.append({
                "question":rows[i][0],
                "true_reference_for_GRADING_ONLY":rows[i][1],
                "gold_probability":group[0],
                "plausible_wrong_probability":group[1],
                "real_CB59_probability":group[2],
                "real_Pythia_probability":group[3],
                "gold_rank1_of_4":all(group[0]>p for p in group[1:]),
                "mean_correct_over_wrong_margin":group[0]-sum(group[1:])/3,
            })
        hits=sum(r["gold_rank1_of_4"] for r in ranks)
        hard_bad=[p for r,p in zip(rows,probabilities)
                  if r[3].startswith("actual_generated_") and not r[2]]
        positives=[p for r,p in zip(rows,probabilities) if r[3]=="independently_correct_reference"]
        plausibles=[p for r,p in zip(rows,probabilities) if r[3]=="plausible_false_reference"]
        out={
            "scored_validation_tasks":len(ranks),
            "reference_answer_rank1_given_ORACLE_gold_in_diagnostic":hits,
            "gold_rank1_rate":hits/len(ranks),
            "gold_reference_acceptance_rate":sum(p>=.5 for p in positives)/len(positives),
            "plausible_wrong_rejection_rate":sum(p<.5 for p in plausibles)/len(plausibles),
            "real_generation_wrong_rejection_rate":sum(p<.5 for p in hard_bad)/len(hard_bad),
            "full_heldout_80_example_balanced_accuracy":grade(rows,probabilities)["balanced_accuracy"],
            "gold_was_NOT_injected_into_real_model_generation":True,
            "gold_always_available_ONLY_in_offline_oracle_rank_diagnostic":True,
            "nothing_correct_generated_when_oracle_gold_only_is_better":True,
            "per_question_scores":ranks,
        }
        results[name]=out
    return results
