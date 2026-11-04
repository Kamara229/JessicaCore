"""No TEST data loaded by unit tests; independent synthetic scoring oracle."""
import unittest
from types import SimpleNamespace

import torch
from v146.protocol import (validate,ARMS,DOMAINS,COUNT,START,END,
    SCI_Q_LABELS,BOOL_Q_LABELS,SELECTED_GAINS)
from v146.sources import sci_questions,bool_questions
from v146.scoring import label_tokens,measure_one,aggregate,summary


class FakeTokenizer:
    def encode(self,text,add_special_tokens=False):
        if add_special_tokens:raise AssertionError("unexpected special tokens")
        codes={" A":11," B":12," C":13," D":14," Yes":15," No":16}
        return [codes[text]] if text in codes else [1,2]


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,input_ids,use_cache=False):
        logits=torch.full((1,input_ids.shape[1],20),-7.0)
        logits[0,-1,12]=3.0
        logits[0,-1,11]=1.0
        return SimpleNamespace(logits=logits)


class FrozenValidChoiceTests(unittest.TestCase):
    def test_fixed_validation_constraints(self):
        self.assertTrue(validate())
        self.assertEqual((START,END,COUNT),(200,208,8))
        self.assertEqual(DOMAINS,("SciQ","BoolQ"))
        self.assertEqual(ARMS,("FP32","CB59","CB140_full","CB140_head_half"))
        self.assertEqual(SELECTED_GAINS["CB140_head_half"],(0.,.5))

    def test_fixed_sciq_correct_label_rotates_without_gold_only_prompt(self):
        cases=[]
        for i in range(COUNT):
            cases.append({
                "question":f"Which answer is correct in case {i}?",
                "correct_answer":"good","distractor1":"bad1",
                "distractor2":"bad2","distractor3":"bad3","support":"some evidence",
            })
        tasks=sci_questions(cases)
        self.assertEqual(len(tasks),COUNT)
        for i,task in enumerate(tasks):
            self.assertEqual(task.correct_index,(START+i)%4)
            self.assertEqual(task.options,("A","B","C","D"))
            self.assertIn("Reply with only",task.prompt)
            self.assertEqual(len(task.prompt_sha),64)

    def test_boolq_validation_bool_label_correct(self):
        cases=[{"question":f"Is number {i} positive?","passage":"a recorded passage",
                "answer":i%2==0} for i in range(COUNT)]
        tasks=bool_questions(cases)
        self.assertEqual([t.correct_index for t in tasks],
                         [0,1,0,1,0,1,0,1])
        self.assertTrue(all(t.options==("Yes","No") for t in tasks))

    def test_fair_next_label_likelihood_on_same_token_set(self):
        ids=label_tokens(FakeTokenizer(),SCI_Q_LABELS)
        self.assertEqual(ids,(11,12,13,14))
        answer=measure_one(FakeModel(),torch.tensor([[1,2,3]]),ids,1)
        self.assertTrue(answer["is_correct"])
        self.assertGreater(answer["confidence_of_correct_label_conditional"],.8)
        miss=measure_one(FakeModel(),torch.tensor([[1,2,3]]),ids,0)
        self.assertFalse(miss["is_correct"])
        self.assertLess(miss["gold_margin_logit_vs_other"],0)

    def test_prevent_silent_multi_token_or_duplicate_labels(self):
        class Bad:
            def encode(self,*args,**kwargs):return [1,2]
        with self.assertRaises(ValueError):
            label_tokens(Bad(),SCI_Q_LABELS)
        with self.assertRaises(ValueError):
            label_tokens(FakeTokenizer(),("A","A"))

    def test_report_preserves_negative_results(self):
        def record(correct,chosen):
            return {"is_correct":correct,"selected_index":chosen,
                    "confidence_of_correct_label_conditional":.2,
                    "gold_margin_logit_vs_other":-1.}
        results={arm:{domain:[record(True,0) for _ in range(COUNT)]
                      for domain in DOMAINS} for arm in ARMS}
        results["CB140_head_half"]["SciQ"]=[record(False,1) for _ in range(COUNT)]
        report=summary(results)
        self.assertFalse(report["CB140_half_no_regression_in_both"])
        self.assertEqual(report["validation"]["SciQ"]
                         ["CB140_half_vs_CB59_accuracy_delta"],-1)
        with self.assertRaises(ValueError):
            aggregate([])


if __name__=="__main__":
    unittest.main()
