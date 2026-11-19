"""S118 no B-GOLD read in synthetic tests; one-shot bootstrap and SHA ordering contract."""
import ast
from pathlib import Path
import unittest
import torch

from research.calibri_nra3.s118_ONE_SHOT_SynTagRusB_NC_paired_trust_external_firstread import (
    STAGE,SEED,BOOT,REPORT,bootstrap_paired_rows,
    S117_SOURCE,S117_MANIFEST,S112_BASE,S115_TRAINED,S112_MANIFEST,
)
from research.calibri_nra3.s118_SynTagRusB_NC_GOLD_AFTER_S117_AND_weightsSHA import (
    SOURCE_SHA,GOLD_SHA,UPSTREAM_BLOB,
)
from research.calibri_nra3.s112_in_calibri_bounded_structural_trust_53751 import (
    NRA3BoundedStructuralTrustInCalibriWeights,
)

class TestS118ProspectiveBNoLeak(unittest.TestCase):
    def test_burned_one_shot_source_SHA_teacher_gold_and_NC(self):
        self.assertIn('S118',STAGE)
        self.assertEqual(SOURCE_SHA,'f687b1f677a3795ba4c4dbf3b5246e1105cf19f236f023e4d9c4d18b5418f9bf')
        self.assertEqual(GOLD_SHA,'fca74d9f7b83cc93374dab774d30dc0d881b62c9a5dcef00a9b113598b4c21d3')
        self.assertEqual(UPSTREAM_BLOB,'416fcb93e223869e2718be30629dad9fadd4c290')
        self.assertEqual(S117_SOURCE,'NRA3_S117_SYNTAGRUS_TRAINB256_ALL12_ORIGINAL_H320_NO_GOLD.safetensors')
        self.assertEqual(S117_MANIFEST,'NRA3_S117_REAL43MGT_SYNTAGRUS_TRAINB256_ALLVERBS_SOURCE_PROVENANCE.json')
        self.assertEqual(S112_BASE,'NRA3_S112_ORIGINAL12_BOUNDED_53751_IN_CALIBRI_ZERO_TRUST_FP16.safetensors')
        self.assertEqual(S115_TRAINED,'NRA3_S117_IN_CALIBRI_TRAINED_TRUST_53751_FP16.safetensors')
        self.assertTrue(REPORT.endswith('.json'))
        self.assertEqual(BOOT,4000)

    def test_paired_sentence_bootstrap_no_unit_as_arg_edges(self):
        # Same accuracy can hide large paired variance. Sample intact sentences.
        per_row=[]
        for i in range(100):
            n=2 if i%3==0 else 1
            base=1 if i%4==0 else 0
            new=1 if i%5==0 else 0
            per_row.append({'role_slots':n,'base_correct':base,'trained_correct':new})
        a=bootstrap_paired_rows(per_row)
        b=bootstrap_paired_rows(per_row)
        self.assertEqual(a,b)
        self.assertEqual(a['n_original_source_sentence_clusters'],100)
        self.assertEqual(a['n_teacher_anchored_supervised_predicate_role_slots'],
                         sum(x['role_slots'] for x in per_row))
        self.assertAlmostEqual(a['paired_delta_percentage_points'],
                               100*(20-25)/sum(x['role_slots'] for x in per_row))
        self.assertLessEqual(a['paired_sentence_cluster_bootstrap_95pct_delta_pp'][0],
                             a['paired_sentence_cluster_bootstrap_95pct_delta_pp'][1])

    def test_bounded_trust_synthetic_old_vs_learned_no_B_gold(self):
        old=NRA3BoundedStructuralTrustInCalibriWeights()
        new=NRA3BoundedStructuralTrustInCalibriWeights()
        with torch.no_grad():
            new.structural_trust_gate.fill_(2.0)
        mask=torch.ones((1,3),dtype=torch.bool)
        base=torch.tensor([5.,0.,-1.]).reshape(1,1,1,3).expand(1,8,3,3).contiguous()
        raw=torch.tensor([-10.,3.,4.]).reshape(1,1,1,3).expand(1,8,3,3).contiguous()
        a=old.bounded_role_competition(base,raw,mask)
        b=new.bounded_role_competition(base,raw,mask)
        self.assertTrue(torch.equal(a,base))
        self.assertTrue(bool(((b-a).abs()<=1.00001).all()))
        self.assertTrue(bool((b.argmax(-1)==a.argmax(-1)).all()))
        self.assertEqual(old.output_gate,new.output_gate)

    def test_raw_source_and_B_GOLD_only_after_PINNED_S112_S115_neural_weights(self):
        src=Path('research/calibri_nra3/s118_ONE_SHOT_SynTagRusB_NC_paired_trust_external_firstread.py').read_text()
        self.assertLess(src.index('require(sha(s115_child)==source_info['),
                        src.index('original,aligned,audit=load_source_VERIFY_sha_THEN_OPEN_gold('))
        self.assertLess(src.index('require(set(old)==set(new)'),
                        src.index('original,aligned,audit=load_source_VERIFY_sha_THEN_OPEN_gold('))
        for x in ("first_and_ONLY_prospective_SynTagRus_B_firstread_SPENT_AFTER_THIS",
                  "NO_dev_or_B_gold_training_or_model_selection_or_repeats",
                  "S115_same_A_internal_developer_NEVER_SCORED",
                  "candidate_accepted",
                  "NONCOMMERCIAL_only_NEVER_deploy_without_rightsholder_license"):
            self.assertIn(x,src)
        imported=[z.module or "" for z in ast.walk(ast.parse(src))
                  if isinstance(z,ast.ImportFrom)]
        self.assertFalse(any("s115_original12_SynTagRusNC_ONLY8" in x
                             or "s110_true12_GSD_16656" in x for x in imported))
        self.assertNotIn("optimizer.step",src)
        self.assertNotIn("loss.backward()",src)

if __name__=="__main__":
    unittest.main()
