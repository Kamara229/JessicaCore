"""S19f source-only synthetic interface checks: NOT real43M quality results."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import inspect
import unittest

import torch

from research.nra3_s19.graph_schema import raw_model_input
from research.nra3_s19.s19e_original43m_raw_alignment import s19_train_texts
from research.nra3_s19.s19f_real43m_train9_graph_targets import (
    STATUS, ROLE, ARG_POLARITY, EVENT_RELATION,
    FrozenModelInputs, assemble_gold_blind_338,
    make_train9_targets, train9_source_manifest, JoinedGraphTrain9,
)
from research.nra3_s19.train_only_event_graph_source import TRAIN_ONLY_FROZEN_SHA256


def synthetic_original_geometry_only():
    """Synthetic test geometry only; never treated as original model hidden."""
    texts=s19_train_texts()
    lengths=[len(raw_model_input(t).tokens) for t in texts]
    max_words=max(lengths)
    hidden=torch.zeros(9,max_words,320,dtype=torch.float16)
    report={"wordpiece_alignment":[], "true_train9_token_counts":lengths}
    for i,n in enumerate(lengths):
        for j in range(n):
            hidden[i,j,:]=float((i+1)*10+j)
        report["wordpiece_alignment"].append({
            "word_last_token_positions":list(range(n)),
            "word_token_positions":[[j] for j in range(n)],
        })
    return texts,hidden,report


class GraphSupervisionOnly(unittest.TestCase):
    def test_physical_interface_is_gold_blind_and_binary_raw18(self):
        texts,h,report=synthetic_original_geometry_only()
        x=assemble_gold_blind_338(texts,h,report)
        self.assertEqual(x.raw_word_hidden320_and_raw18.shape[0],9)
        self.assertEqual(x.raw_word_hidden320_and_raw18.shape[-1],338)
        self.assertEqual(int(x.valid_word_mask.sum()),
                         sum(len(raw_model_input(t).tokens) for t in texts))
        self.assertTrue(bool(((x.raw_word_hidden320_and_raw18[...,320:]==0)|
                              (x.raw_word_hidden320_and_raw18[...,320:]==1)).all()))
        self.assertFalse(hasattr(x,"event_status"))
        self.assertFalse(hasattr(x,"roles"))
        self.assertFalse(hasattr(x,"coreference"))
        self.assertNotIn("graph",inspect.signature(assemble_gold_blind_338).parameters)
        self.assertTrue(bool((x.raw_word_hidden320_and_raw18[~x.valid_word_mask]==0).all()))

    def test_gold_event_graph_has_all_five_supervision_targets_in_separate_bundle(self):
        texts,h,report=synthetic_original_geometry_only()
        x=assemble_gold_blind_338(texts,h,report)
        y=make_train9_targets(x)
        self.assertEqual(int((y.event_status>0).sum()),15)
        self.assertEqual(int((y.predicate_argument_role>0).sum()),40)
        self.assertEqual(int((y.argument_polarity==1).sum()),1)
        self.assertEqual(int((y.coreference==1).sum()),5)
        self.assertEqual(int((y.event_links>0).sum()),6)
        self.assertTrue(bool((y.event_status[~x.valid_word_mask]==-1).all()))

    def test_negated_event_vs_rejected_argument_recipients_separate(self):
        texts,h,report=synthetic_original_geometry_only()
        x=assemble_gold_blind_338(texts,h,report)
        y=make_train9_targets(x)
        p=raw_model_input(texts[6]).tokens
        pos={t.form:t.index-1 for t in p}
        self.assertEqual(y.event_status[6,pos["подписал"]].item(),
                         STATUS.index("ASSERTED"))
        self.assertEqual(y.event_status[6,pos["отправила"]].item(),
                         STATUS.index("NEGATED"))
        tokens=raw_model_input(texts[7]).tokens
        pos={t.form:t.index-1 for t in tokens}
        event=pos["отправила"]
        self.assertEqual(y.event_status[7,event].item(),STATUS.index("ASSERTED"))
        for recipient,polarity in [("клиенту","REJECTED"),("юристу","POSITIVE")]:
            j=pos[recipient]
            self.assertEqual(y.predicate_argument_role[7,event,j].item(),
                             ROLE.index("RECIPIENT"))
            self.assertEqual(y.argument_polarity[7,event,j].item(),
                             ARG_POLARITY.index(polarity))

    def test_conditional_vs_temporal_are_distinct_events_and_relations(self):
        texts,h,report=synthetic_original_geometry_only()
        x=assemble_gold_blind_338(texts,h,report)
        y=make_train9_targets(x)
        for k,expected in ((4,"CONDITION_RESULT"),(5,"TEMPORAL_SEQUENCE")):
            ts=raw_model_input(texts[k]).tokens
            pos={t.form:t.index-1 for t in ts}
            p,q=pos["подпишет"],pos["отправит"]
            self.assertEqual(y.event_links[k,p,q].item(),
                             EVENT_RELATION.index(expected))
            if k==4:
                self.assertEqual(y.event_status[k,p].item(),
                                 STATUS.index("CONDITIONAL"))
                self.assertEqual(y.event_status[k,q].item(),
                                 STATUS.index("CONDITIONAL"))
            else:
                self.assertEqual(y.event_status[k,p].item(),
                                 STATUS.index("FUTURE_UNREALIZED"))

    def test_bad_raw_alignment_or_text_fails_closed(self):
        texts,h,report=synthetic_original_geometry_only()
        with self.assertRaises(ValueError):
            assemble_gold_blind_338(texts[:-1]+("новый чужой текст",),h,report)
        bad=deepcopy(report)
        bad["wordpiece_alignment"][0]["word_token_positions"][1]=[0]
        with self.assertRaises(ValueError):
            assemble_gold_blind_338(texts,h,bad)
        a=assemble_gold_blind_338(texts,h,report)
        with self.assertRaises(ValueError):
            make_train9_targets(replace(a,raw_texts=a.raw_texts[::-1]))

    def test_gold_and_hidden_source_digest_contract_without_claiming_model_result(self):
        texts,h,report=synthetic_original_geometry_only()
        a=assemble_gold_blind_338(texts,h,report)
        j=JoinedGraphTrain9(a,make_train9_targets(a),
                             TRAIN_ONLY_FROZEN_SHA256,"synthetic-only",True)
        m=train9_source_manifest(j)
        self.assertEqual(m["train_only_sentences"],9)
        self.assertEqual(m["supervised_events"],15)
        self.assertEqual(m["REJECTED_argument_links"],1)
        self.assertTrue(m["no_optimizer_no_new_model_scoring"])
        self.assertTrue(m["all_human_gold_event_role_scope_coref_relations_target_only"])
        self.assertFalse(m["activation_allowed"])


if __name__=="__main__":
    unittest.main()
