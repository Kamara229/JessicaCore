"""Regression: frozen 16+16 fresh QA, negative outcomes must stay visible."""
import unittest
import torch

from v148.protocol import (verify_protocol,DOMAINS,ARMS,COUNT,START,END,
                           FROZEN_CB147_SHA)
from v148.sources import make_sciq,make_boolq
from v148.scoring import make_score,audit


def record(row,chosen,good=True):
    return {
        "row":row,"prompt_sha":f"frozen_hash_for_row_{row}",
        "chosen_index":chosen,"correct":good,
        "gold_conditional_probability":.2,
        "gold_minus_best_other_logit":-2,
        "conditional_choice_probabilities":[.2,.8],
    }


class FrozenPaidPriorQA(unittest.TestCase):
    def test_source_predeclared_and_distinct_from_v146(self):
        self.assertTrue(verify_protocol())
        self.assertEqual((START,END,COUNT),(400,416,16))
        self.assertEqual(len(ARMS),4)
        self.assertEqual(DOMAINS,("SciQ","BoolQ"))
        self.assertEqual(len(FROZEN_CB147_SHA),64)
        self.assertNotIn(200,range(START,END))

    def test_actual_labels_are_only_scoring_indices(self):
        synthetic=[
            {"question":f"Which source answer for {j}?",
             "correct_answer":"good","distractor1":"wrong1",
             "distractor2":"wrong2","distractor3":"wrong3",
             "support":"source-only support"} for j in range(COUNT)]
        questions=make_sciq(synthetic)
        self.assertEqual(len(questions),16)
        self.assertEqual(questions[0].index,400)
        self.assertEqual(questions[-1].index,415)
        self.assertEqual(questions[7].correct_index,3)
        self.assertTrue(all(q.options==("A","B","C","D") for q in questions))

        bool_rows=[{"passage":"trusted content", "question":"Is it true?",
                    "answer":(k%2)==0} for k in range(COUNT)]
        tasks=make_boolq(bool_rows)
        self.assertEqual(tasks[0].options,("Yes","No"))
        self.assertEqual(tasks[1].correct_index,1)

    def test_normalized_true_choice_prob_and_wrong_answer_preserved(self):
        logits=torch.tensor([0.,-1.,3.,2.,-2.,-5.])
        yes=make_score(logits,[2,3],0)
        no=make_score(logits,[2,3],1)
        self.assertTrue(yes["correct"])
        self.assertFalse(no["correct"])
        self.assertAlmostEqual(sum(yes["conditional_choice_probabilities"]),1,places=6)
        with self.assertRaises(ValueError):
            make_score(torch.ones(2),[0,1],3)

    def test_regression_per_source_cannot_be_hidden(self):
        data={arm:{d:[record(START+i,1,good=True) for i in range(COUNT)]
                   for d in DOMAINS} for arm in ARMS}
        data["CB147_int8_paid"]["SciQ"]=[
            record(START+i,0,good=False) for i in range(COUNT)]
        report=audit(data)
        self.assertFalse(report["CB147_no_choice_accuracy_regression_in_both"])
        self.assertEqual(report["fixed_validation_only"]["SciQ"]
                         ["CB147_minus_CB59_correct"],-16)
        self.assertEqual(report["fixed_validation_only"]["BoolQ"]
                         ["CB147_minus_CB59_correct"],0)
        data["CB147_int8_paid"]["SciQ"].pop()
        with self.assertRaises(ValueError):
            audit(data)


if __name__=="__main__":
    unittest.main()
