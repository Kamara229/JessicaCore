"""No public QA data fetched; strict fixed semantic validation guard tests."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from v163.protocol import (check,ARMS,PACKED_ARMS,DATASETS,
    ROW_BEGIN,ROW_END,COUNT,GENERATION_INDICES)
from v163.sources import questions_sciq,questions_boolq
from v163.audit import summarize_choices,summarize_generation


def choice(correct=True,index=0,gold=.3):
    return {"chosen_index":index,"correct":correct,
            "gold_conditional_probability":gold,
            "gold_minus_best_other_logit":0.1}


class PaidBlock0SemanticTruthTest(unittest.TestCase):
    def test_fixed_fresh_official_VALID_ranges_and_actual_generation(self):
        self.assertTrue(check())
        self.assertEqual((ROW_BEGIN,ROW_END,COUNT),(704,720,16))
        self.assertEqual(GENERATION_INDICES,(2,5,8,12))
        self.assertEqual(DATASETS,("SciQ","BoolQ"))
        self.assertEqual(PACKED_ARMS,("original_CB59","fully_paid_CB161_block0"))

    def test_qa_answers_are_not_learned_and_match_across_all_three_arms(self):
        cases=[]
        for j in range(COUNT):
            current=ROW_BEGIN+j
            cases.append({"index":current,
                         "prompt_sha":hashlib.sha256(str(current).encode()).hexdigest()})
        data={}
        for arm in ARMS:
            data[arm]={}
            for domain in DATASETS:
                records=[]
                for x in cases:
                    records.append({
                       **choice(correct=True,index=1),
                       "row":x["index"],"prompt_sha":x["prompt_sha"]})
                data[arm][domain]=records
        final=summarize_choices(data)
        self.assertTrue(final["no_2domain_forced_choice_regression"])
        data[ARMS[-1]][DATASETS[1]][-1]["correct"]=False
        revised=summarize_choices(data)
        self.assertFalse(revised["no_2domain_forced_choice_regression"])
        self.assertEqual(revised["fresh_science_boolean_question_accuracy"][
            DATASETS[1]]["paid_block0_minus_original_CB59_correct"],-1)
        data[ARMS[0]][DATASETS[0]].pop()
        with self.assertRaises(ValueError):
            summarize_choices(data)

    def test_actual_sciq_and_boolq_original_labels_and_option_integrity(self):
        sci={
            "correct_answer":"hydrogen","distractor1":"oxygen",
            "distractor2":"helium","distractor3":"neon",
            "question":"Name the lightest gas?",
            "support":"Hydrogen is the lightest element."}
        expected=questions_sciq([sci for _ in range(COUNT)])
        self.assertEqual(len(expected),16)
        self.assertEqual([x.index for x in expected],
                         list(range(ROW_BEGIN,ROW_END)))
        self.assertTrue(all(f"{'ABCD'[c.correct_index]}. hydrogen" in c.prompt
                            for c in expected))
        bool_cases=questions_boolq([
            {"question":"Is water wet?","passage":"Water is a liquid.","answer":True}
            for _ in range(COUNT)])
        self.assertTrue(all(x.correct_index==0 for x in bool_cases))
        with self.assertRaises(ValueError):
            questions_sciq([{**sci,"distractor3":"oxygen"}])

    def test_full_greedy_require_same_four_prompts_and_true_token_counts(self):
        records={}
        for arm in PACKED_ARMS:
            records[arm]=[]
            for i in GENERATION_INDICES:
                ids=[i,i+1,i]
                records[arm].append({
                    "prior_historical_prompt_index":i,
                    "actual_full_generated_token_ids":ids,
                    "EOS_generated":False,"full_model_wall_seconds":1.,
                })
        real=summarize_generation(records)
        self.assertEqual(real["genuine_packed_tokens_per_second_ratio_paid_vs_CB59"],1.)
        self.assertEqual(real["genuine_full_model_autoregressive_measures"][
            PACKED_ARMS[0]]["actual_generated_token_count"],12)
        records[PACKED_ARMS[1]][-1]["prior_historical_prompt_index"]=99
        with self.assertRaises(ValueError):
            summarize_generation(records)


if __name__=="__main__":
    unittest.main()
