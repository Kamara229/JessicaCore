"""S19b source/gradient/FP16 tests only. NO independent EVAL or LIVE."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

import torch

from research.nra3_s0.neural_area import UPOS
from research.nra3_s19.paid_grammar_parser import (
    CompactWordGrammar, GrammarLogits, DEP_LABELS,
    INPUT_DIM, RANK, paid_fp16_bytes, load_paid_fp16_bytes,
    trainable_parameter_count,
)
from research.nra3_s19.s19b_train96_true43m_grammar import (
    WordGrammarTrainData, assemble_true43m_train96, combine_results,
    metrics, word_loss,
)
from research.nra3_s18.ru96_multiverb_train_source import build_s18_new_train96


def fake_training_batch():
    torch.manual_seed(19018)
    raw = torch.randn((3, 5, INPUT_DIM), dtype=torch.float32)
    valid = torch.tensor([[1,1,1,0,0], [1,1,1,1,0],
                          [1,1,1,1,1]], dtype=torch.bool)
    raw[~valid] = 0.
    upos = torch.tensor([[UPOS.index("PROPN"),UPOS.index("VERB"),UPOS.index("NOUN"),-1,-1],
                        [UPOS.index("PROPN"),UPOS.index("VERB"),UPOS.index("NOUN"),UPOS.index("PUNCT"),-1],
                        [UPOS.index("ADV"),UPOS.index("PROPN"),UPOS.index("VERB"),UPOS.index("NOUN"),UPOS.index("PUNCT")]])
    dep = torch.tensor([[1,0,3,-1,-1],[1,0,3,4,-1],[10,1,0,3,4]])
    head = torch.tensor([[2,0,2,-1,-1],[2,0,2,2,-1],[3,3,0,3,3]])
    pred = torch.tensor([[0,1,0,-1,-1],[0,1,0,0,-1],[0,0,1,0,0]])
    return WordGrammarTrainData(raw,valid,upos,dep,head,pred,"test-train","test-hidden")


class PaidGrammarTrainContracts(unittest.TestCase):
    def test_shapes_all_words_and_root_vs_self_arc(self):
        model=CompactWordGrammar()
        batch=fake_training_batch()
        got=model(batch.input_338,batch.valid)
        self.assertEqual(got.upos.shape,(3,5,len(UPOS)))
        self.assertEqual(got.dependency.shape,(3,5,len(DEP_LABELS)))
        self.assertEqual(got.predicate.shape,(3,5))
        self.assertEqual(got.heads.shape,(3,5,6))
        for i in range(3):
            self.assertTrue(torch.all(got.heads[i,range(5),1+torch.arange(5)] == -1e4))
            self.assertTrue(torch.isfinite(got.heads[i]).all())
        self.assertTrue(bool((got.heads[0,:,:][:,4:] == -1e4).all()))

    def test_loss_backprop_trains_every_paid_parameter(self):
        batch=fake_training_batch()
        model=CompactWordGrammar()
        loss=word_loss(model(batch.input_338,batch.valid),batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        self.assertGreater(trainable_parameter_count(model),0)

    def test_all_physical_fp16_weights_included_and_round_trip(self):
        model=CompactWordGrammar().eval()
        blob=paid_fp16_bytes(model)
        loaded=load_paid_fp16_bytes(blob)
        self.assertEqual(blob,paid_fp16_bytes(loaded))
        self.assertTrue(all(t.dtype==torch.float32 for t in loaded.state_dict().values()))
        self.assertTrue(all(not t.requires_grad for t in loaded.parameters()))
        self.assertEqual(len(hashlib.sha256(blob).hexdigest()),64)
        self.assertGreater(len(blob),trainable_parameter_count(model)*2)

    def test_gold_blind_input_model_rejects_wrong_shape_or_activation(self):
        batch=fake_training_batch()
        model=CompactWordGrammar()
        with self.assertRaises(ValueError):
            model(batch.input_338[..., :320],batch.valid)
        with self.assertRaises(ValueError):
            model(batch.input_338,batch.valid.float())
        with self.assertRaises(ValueError):
            model(batch.input_338,~batch.valid)
        model.output_gate=1
        with self.assertRaises(ValueError):
            model(batch.input_338,batch.valid)
        with self.assertRaises(ValueError):
            paid_fp16_bytes(model)

    def test_internal_aggregates_are_count_weighted_not_fold_mean(self):
        a={"tokens":10,"upos_correct":5,"dependency_label_correct":7,
           "head_correct":4,"las_correct":3,"predicate_TP":2,
           "predicate_FP":1,"predicate_FN":2}
        b={"tokens":2,"upos_correct":2,"dependency_label_correct":2,
           "head_correct":2,"las_correct":1,"predicate_TP":1,
           "predicate_FP":1,"predicate_FN":0}
        m=combine_results([a,b])
        self.assertEqual(m["tokens"],12)
        self.assertAlmostEqual(m["upos_accuracy"],7/12)
        self.assertAlmostEqual(m["head_UAS"],6/12)
        self.assertAlmostEqual(m["exact_head_and_deprel_LAS"],4/12)
        self.assertAlmostEqual(m["predicate_micro_F1"],6/(6+2+2))

    def test_micro_metrics_on_inference_only_roundtrip(self):
        b=fake_training_batch()
        model=load_paid_fp16_bytes(paid_fp16_bytes(CompactWordGrammar()))
        record=metrics(model,b)
        self.assertEqual(record["tokens"],12)
        self.assertEqual(record["predicate_TP"]+record["predicate_FN"],3)
        self.assertTrue(0. <= record["exact_head_and_deprel_LAS"] <= 1.)

    def test_train96_pos_deprel_vocabulary_covers_all_96_source_only(self):
        rows=build_s18_new_train96()
        self.assertEqual(len(rows),96)
        labels={word.deprel for row in rows for word in row.words}
        self.assertEqual(labels,set(DEP_LABELS))
        for row in rows:
            for word in row.words:
                self.assertIn(word.upos,UPOS)
                self.assertGreaterEqual(word.head,0)
                self.assertLessEqual(word.head,len(row.words))


if __name__=="__main__":
    unittest.main()
