"""Z42 fail-closed synthetic contracts for read-only archived 43M physical audit."""
import copy
import unittest
from research.calibri_nra3.s119q2i8z42_z41_spent14_native_entropy_causal_audit import (
    BASE,TRUE,WRONG,MODES,inspect_arm,empirical,sign_p,audit
)

def fixture():
    rows=[]
    counts={m:{"correct_FIRST":0,"correct_FIRST_TWO":0,
               "correct_FIRST_THREE":0,"correct_FIRST_FOUR":0,
               "correct_FIRST_SIX":0,
               "distinct_second_native_BPE_ids":0,
               "premature_EOS_in_first4":0} for m in MODES}
    seconds={m:[] for m in MODES}
    for i in range(14):
        gold=[7,8,9,10]
        x={BASE:8 if i==0 else 100+i,
           TRUE:8 if i in (0,1) else 200+i%3,
           WRONG:300+i%7}
        modes={}
        for m in MODES:
            ids=[7,x[m],700+i,800+i,900+i,901+i]
            second_correct=ids[:2]==gold[:2]
            steps=[]
            for pos in range(2,7):
                chosen=ids[pos-1]
                greedy=(x[BASE] if pos==2 else chosen)
                corrected=m!=BASE and pos==2
                steps.append({
                    "native_position_1based":pos,
                    "chosen_native_id":chosen,
                    "original_greedy_native_id":greedy,
                    "shadow_changed_original_greedy":chosen!=greedy,
                    "candidate_original_top64":True,
                    "chose_EOS":False,
                    "targetblind_entropy_guard_alpha":0.5 if corrected else 0.,
                    "targetblind_entropy_vs_original_bits":-.01 if corrected else 0.,
                })
            modes[m]={
                "autonomous_first_six_native_ids":ids,
                "first_native_correct":True,
                "first_two_native_correct":second_correct,
                "first_three_native_correct":False,
                "first_four_native_correct":False,
                "first_six_native_correct":False,
                "true_full_S142_native_forwards":6,
                "did_emit_EOS_early":False,
                "per_second_to_fourth_real_forward":steps}
            seconds[m].append(ids[1])
            counts[m]["correct_FIRST"]+=1
            counts[m]["correct_FIRST_TWO"]+=int(second_correct)
        rows.append({"source_id":f"TEST_SOURCE_{i}",
                     "prompt_SHA256":f"TEST_UNIQUE_SHA_{i}",
                     "human_gold_native_first4_POSTHOC":gold,
                     "auto_generated_modes_no_teacher_forcing":modes,
                     "actual_original43M_full_native_forwards":16,
                     "DEV_gold_was_never_in_training_or_candidate_selection":True,
                     "NEW_Z40_DEV_NOW_SPENT_internal_synthetic":True})
    for m in MODES:
        counts[m]["distinct_second_native_BPE_ids"]=len(set(seconds[m]))
    return {"NEW_DEV14_per_prompt_first6":rows,
       "NEW_DEV14_full_autonomous_first6_native":{
         "original43M_real_native_TRAIN_forwards":252,
         "brand_NEW_synthetic_DEV14":14,
         "no_DEV_teacher_forcing":True,
         "only_native_BPE2_BPE3_BPE4_can_be_reranked":True,
         "native_BPE5_BPE6_unchanged_ON_policy":True,
         "targetblind_native_entropy_and_KL_guard":True,
         "original43M_real_native_DEV_forwards":224,
         "matched_autonomous_first6_native_generated":counts}}


class TestZ42SourceOnlyAudit(unittest.TestCase):
    def test_paired_comparison_position_3_and_cross_prompt_diversity(self):
        result=inspect_arm(fixture(),"synthetic")
        base=result["scores"][BASE]
        true=result["scores"][TRUE]
        wrong=result["scores"][WRONG]
        self.assertEqual((base["correct_first2"],true["correct_first2"],
                          wrong["correct_first2"]),(1,2,0))
        self.assertEqual((base["correct_first3"],true["correct_first3"]),(0,0))
        self.assertEqual(true["first2_correct_but_BPE3_wrong"],2)
        self.assertEqual(true["paired_vs_original"]["first2"]["new_correct"],1)
        self.assertEqual(true["per_position_actual_native_interventions"]["2"]["attempts"],14)
        self.assertEqual(true["per_position_actual_native_interventions"]["2"]["alpha_nonzero"],14)
        self.assertEqual(true["per_position_actual_native_interventions"]["3"]["alpha_nonzero"],0)
        self.assertEqual(base["empirical_cross_prompt_native_ID_diversity"]["2"]["distinct"],14)
        self.assertEqual(true["empirical_cross_prompt_native_ID_diversity"]["2"]["distinct"],4)
    def test_entropy_is_empirical_not_the_model_probability_entropy(self):
        self.assertEqual(empirical([1,2,3,4])["empirical_entropy_bits"],2.)
        self.assertEqual(empirical([1,1,1,1])["empirical_entropy_bits"],0.)
        self.assertEqual(sign_p(3,0),.25)
        self.assertEqual(sign_p(0,0),1.)
    def test_reject_modified_unprotected_BPE5_or_forbidden_alpha(self):
        e=fixture()
        r=e["NEW_DEV14_per_prompt_first6"][0]["auto_generated_modes_no_teacher_forcing"][TRUE]
        r["per_second_to_fourth_real_forward"][3]["original_greedy_native_id"]=9001
        with self.assertRaises(ValueError):inspect_arm(e,"bad BPE5")
        e=fixture()
        r=e["NEW_DEV14_per_prompt_first6"][0]["auto_generated_modes_no_teacher_forcing"][TRUE]
        r["per_second_to_fourth_real_forward"][0]["targetblind_entropy_guard_alpha"]=.777
        with self.assertRaises(ValueError):inspect_arm(e,"bad alpha")
    def test_fail_closed_on_wrong_source_or_unsubstantiated_forwards(self):
        with self.assertRaises((KeyError,ValueError)):
            audit({"stage":"wrong","output_gate":0,"candidate_accepted":False})
        e=fixture()
        e["NEW_DEV14_full_autonomous_first6_native"]["original43M_real_native_DEV_forwards"]=200
        with self.assertRaises(ValueError):inspect_arm(e,"bad paid forwards")

if __name__=="__main__":
    unittest.main()
