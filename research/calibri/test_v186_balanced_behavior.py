"""v186 preregistered balanced pair, heldout split and differentiable loss tests."""
import unittest
from collections import Counter
from types import SimpleNamespace

import torch
from torch import nn
from v186 import protocol as p
from v186.pairs import raw_pairs,build_pairs,manifest
from v186.fit import paired_loss

class ToyTokenizer:
    eos_token_id=2
    def __init__(self):self.vocab={"<pad>":0,"<s>":1,"</s>":2}
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
        if not add_generation_prompt:raise ValueError("Required chat prompt")
        return "<user> "+messages[0]["content"]+" <assistant> "
    def encode(self,text,add_special_tokens=False):
        result=[]
        for token in text.strip().split():
            if token not in self.vocab:self.vocab[token]=len(self.vocab)
            result.append(self.vocab[token])
        return result

class SimpleBehaviorModel(nn.Module):
    def __init__(self,vocab):
        super().__init__()
        self.e=nn.Embedding(vocab,12)
        self.head=nn.Linear(12,vocab)
    def forward(self,input_ids,use_cache=False):
        return SimpleNamespace(logits=self.head(self.e(input_ids)))

class PairedBehaviorRegression(unittest.TestCase):
    def test_exact_balance_and_no_split_or_DEV_leakage(self):
        train=raw_pairs("TRAIN");valid=raw_pairs("VALID")
        self.assertEqual(len(train),24)
        self.assertEqual(len(valid),12)
        self.assertEqual(Counter(x["domain"] for x in train),{k:6 for k in p.TRAIN_DOMAINS})
        self.assertEqual(Counter(x["domain"] for x in valid),{k:3 for k in p.TRAIN_DOMAINS})
        left_train={x["positive"][0] for x in train}|{x["negative"][0] for x in train}
        left_valid={x["positive"][0] for x in valid}|{x["negative"][0] for x in valid}
        self.assertEqual(len(left_train),48)
        self.assertEqual(len(left_valid),24)
        self.assertFalse(left_train & left_valid)
        binary=[a for x in train[:12] for a in (x["positive"][1],x["negative"][1])]
        self.assertEqual(binary.count("Yes"),12)
        self.assertEqual(binary.count("No"),12)
        with self.assertRaises(ValueError):raw_pairs("TEST")
        self.assertEqual(len(p.SCHEDULE),96)
        self.assertEqual(len(set(p.SCHEDULE)),96)
        self.assertEqual(p.SOURCE_BYTES,11229303)
        self.assertLess(p.SOURCE_BYTES,p.MAX_BYTES)

    def test_chat_tokenizer_uses_real_answer_positions_and_eos(self):
        tok=ToyTokenizer()
        train=build_pairs(tok,"TRAIN",())
        valid=build_pairs(tok,"VALID",())
        self.assertEqual(len(manifest(train)),24)
        self.assertEqual(len(manifest(valid)),12)
        for pair in train+valid:
            l,r=pair["left"],pair["right"]
            self.assertNotEqual(l["first_expected_id"],r["first_expected_id"])
            for sample in (l,r):
                self.assertEqual(int(sample["targets"][-1]),tok.eos_token_id)
                self.assertEqual(int(sample["targets"][0]),sample["first_expected_id"])
                self.assertGreater(int(sample["positions"][0]),0)
        with self.assertRaises(AssertionError):
            build_pairs(tok,"TRAIN",[train[0]["left"]["question"]])

    def test_joint_opposing_targets_have_real_finite_gradients(self):
        torch.manual_seed(186)
        tokenizer=ToyTokenizer()
        sample=build_pairs(tokenizer,"TRAIN",())[0]
        model=SimpleBehaviorModel(len(tokenizer.vocab)+5)
        loss,detail=paired_loss(model,None,sample)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(detail["expected_first_tokens"]),2)
        self.assertNotEqual(*detail["expected_first_tokens"])
        self.assertEqual(len(detail["first_token_logit_gaps"]),2)
        loss.backward()
        self.assertTrue(torch.isfinite(model.e.weight.grad).all())
        self.assertGreater(float(model.head.weight.grad.abs().sum()),0)

if __name__=="__main__":
    unittest.main()
