"""Predeclared independent QA rows and immutable physical-source assertions."""
import unittest
import torch
from v150.protocol import (check_protocol,ROW_BEGIN,ROW_END,QUESTION_COUNT,
                            GENERATION_INDICES,ARMS,DATASETS,CB149_SHA)
from v150.sources import make_sciq,make_boolq
from v150.audit import summarize
from v148.scoring import make_score


def example(row,correct=True,choice=0):
    return {
        "question_row":row,
        "prompt_SHA256":f"prelocked_question_{row}",
        "correct":correct,
        "chosen_index":choice,
        "gold_conditional_probability":.25,
        "gold_minus_best_other_logit":-1.,
    }


class ImmutableCB149QA(unittest.TestCase):
    def test_fresh_valid_slice_and_frozen_generation_probes(self):
        self.assertTrue(check_protocol())
        self.assertEqual((ROW_BEGIN,ROW_END,QUESTION_COUNT),(600,624,24))
        self.assertEqual(GENERATION_INDICES,(4,5,8,9,12,13))
        self.assertEqual(len(ARMS),4)
        self.assertEqual(len(CB149_SHA),64)
        self.assertEqual(Datasetsize:=len(DATASETS),2)
        self.assertGreater(ROW_BEGIN,415)

    def test_correct_answer_placement_is_fixed_prior_to_logits(self):
        rows=[{
          "question":f"Science question {i}",
          "correct_answer":"gold",
          "distractor1":"foil1","distractor2":"foil2","distractor3":"foil3",
          "support":"source context only",
        } for i in range(QUESTION_COUNT)]
        result=make_sciq(rows)
        self.assertEqual(len(result),24)
        self.assertEqual(result[0].correct_index,0)
        self.assertEqual(result[5].correct_index,1)
        self.assertTrue(all(item.options==("A","B","C","D") for item in result))
        self.assertEqual(len({r.prompt_sha for r in result}),24)
        rows=[{
          "question":f"Check true? {i}",
          "answer":i%2==0,
          "passage":"some valid paragraph",
        } for i in range(QUESTION_COUNT)]
        parsed=make_boolq(rows)
        self.assertEqual(parsed[0].correct_index,0)
        self.assertEqual(parsed[1].correct_index,1)

    def test_choice_confidence_and_hidden_error_preserved(self):
        logit=torch.tensor([2.,0.,-1.,1.])
        correct=make_score(logit,[0,1,2,3],0)
        wrong=make_score(logit,[0,1,2,3],1)
        self.assertTrue(correct["correct"])
        self.assertFalse(wrong["correct"])
        self.assertAlmostEqual(sum(correct["conditional_choice_probabilities"]),1,places=6)

    def test_per_domain_regression_cannot_be_hidden(self):
        data={arm:{domain:[example(k) for k in range(ROW_BEGIN,ROW_END)]
                   for domain in DATASETS} for arm in ARMS}
        data["CB149_contextual"]["SciQ"]=[
            example(k,correct=False,choice=1) for k in range(ROW_BEGIN,ROW_END)]
        report=summarize(data)
        self.assertFalse(report["contextual_int4_no_accuracy_regression_in_both"])
        self.assertEqual(report["fresh_SciQ_and_BoolQ_validation"]["SciQ"]
                         ["CB149_minus_CB147_correct"],-24)
        data["CB59"]["BoolQ"].pop()
        with self.assertRaises(ValueError):
            summarize(data)


if __name__=="__main__":
    unittest.main()
