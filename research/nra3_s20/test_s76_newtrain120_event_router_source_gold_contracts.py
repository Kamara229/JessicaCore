"""S76 EVENT-relative new TRAIN120 labels and zero-effect original logit ABI."""
import unittest

import torch

from research.nra3_s20.s74_newtrain120_source_ONLY import (
    source_sha256,GROUP_TRAIN,GROUP_DIAG,
)
from research.nra3_s20.s76_s74_newtrain120_event_relative_gold_TRAIN_LOSS_ONLY import (
    template_constructed_OLDTRAIN_event_area_gold120_for_loss_only,preregistration,
)
from research.nra3_s20.s72_nra2_situation_microareas import SituationMicroAreas
from research.nra3_s20.s76_train_newtrain120_true43m_8area_factor_router import (
    area_metrics,SEED,EPOCHS,TRAIN_WEIGHTS,
)


class TestS76Original43mTrainOnlySituationRouting(unittest.TestCase):
    def test_event_relative_target_is_not_utterance_level_negation(self):
        targets=template_constructed_OLDTRAIN_event_area_gold120_for_loss_only()
        self.assertEqual(len(targets),120)
        for n in range(12):
            f=targets[n*10:n*10+10]
            self.assertEqual([x["row_id"] for x in f],
                             [f"T{n+1:02d}_{i:02d}" for i in range(1,11)])
            # Later successful action must not be supervised as negated.
            self.assertEqual(f[6]["event_area_multi_hot_ONLY_in_TRAIN_loss"][1],0)
            self.assertEqual(f[6]["event_area_multi_hot_ONLY_in_TRAIN_loss"][3],1)
            # Earlier actual action must not inherit negation of second.
            self.assertEqual(f[7]["event_area_multi_hot_ONLY_in_TRAIN_loss"][1],0)
            self.assertEqual(f[7]["event_area_multi_hot_ONLY_in_TRAIN_loss"][6],1)
            self.assertTrue(f[6]["main_event_raw_index_ONLY_in_TRAIN_loss"]>0)
        self.assertTrue(all(x["not_in_real43m_or_source_only_features"]
                            for x in targets))

    def test_precommitted_90_and_30_no_independent_eval(self):
        p=preregistration()
        self.assertEqual(p["S74_source_sha256"],source_sha256())
        self.assertEqual(p["n_new_source_rows"],120)
        self.assertEqual(len(GROUP_TRAIN),9)
        self.assertEqual(len(GROUP_DIAG),3)
        self.assertTrue(set(GROUP_TRAIN).isdisjoint(GROUP_DIAG))
        self.assertTrue(p["constructed_targets_not_independently_human_verified"])
        self.assertTrue(p["prospective_independent_real_world_generalization_NOT_claimed"])
        self.assertEqual(SEED,761119)
        self.assertEqual(EPOCHS,50)
        self.assertEqual(TRAIN_WEIGHTS,4232)

    def test_zero_router_training_effect_on_existing_masked_frozen_scores(self):
        torch.manual_seed(15)
        b,k,w=3,4,9
        net=SituationMicroAreas()
        src=torch.randn(b,k,36)
        words=torch.randn(b,w,320).half()
        indices=torch.randint(0,w,(b,k),dtype=torch.long)
        valid=torch.ones(b,k,w,dtype=torch.bool)
        allowed=torch.ones(b,k,dtype=torch.bool)
        valid[0,0,2]=False
        allowed[1,2]=False
        logits=torch.randn(b,k,w+1)
        logits[0,0,2]=-29997.125  # old parent NOT S72 sentinel -1e9
        logits[1,2,:]=-88009.0
        neg=torch.randn(b,k,w)
        factual=torch.randn(b,k,3)
        out=net(src,words,indices,valid,allowed,logits,neg,factual)
        self.assertTrue(torch.equal(out.target_or_none,logits))
        self.assertTrue(torch.equal(out.rejected_edge,neg))
        self.assertTrue(torch.equal(out.event_factuality,factual))
        self.assertFalse(out.activation_allowed)
        self.assertFalse(out.candidate_accepted)

    def test_oracle_event_anchoring_is_diagnostic_not_end_to_end(self):
        labels=torch.zeros(120,8)
        labels[:,0]=1
        router=torch.zeros(120,1,8)
        router[:,:,0]=3.0
        router[:,:,1:]=-3
        slots=torch.zeros(120,dtype=torch.long)
        selected=torch.zeros(120,dtype=torch.bool)
        selected[:30]=True
        score=area_metrics(router,labels,slots,selected)
        self.assertEqual(score["micro_f1"],1.0)
        self.assertEqual(score["scored_teacher_event_source_rows"],30)
        self.assertTrue(score["scored_oracle_event_anchor_not_model_discovery"])
        self.assertTrue(score["forward_uses_no_gold_features"])

if __name__=="__main__":
    unittest.main()
