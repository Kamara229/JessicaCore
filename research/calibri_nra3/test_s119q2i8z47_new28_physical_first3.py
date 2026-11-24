"""S119q2i8z47 prereg Z46 new28 DEV physical causal/provenance guards."""
import unittest
from pathlib import Path
import torch
from research.calibri_nra3 import s119q2i8z46_fresh28_onpolicy_bpe2_preregister as source
from research.calibri_nra3 import s119q2i8z47_fresh28_native_bpe2_control_eval as model

DIR=Path(__file__).resolve().parent
PHYSICAL=DIR/"s119q2i8z47_physical_s142_new28_dev_bpe2_first3.py"

class TestZ47Fresh28(unittest.TestCase):
    def test_prereg_synthetic_author_dev_is_new_and_sha_unchanged(self):
        v=source.verify()
        self.assertEqual(v["brand_NEW_28_internal_DEV_SHA256"],
            "c4489970fbb0868f47c15fe6cc47b348ec4c533e094e0775f2275e00bd4bc58c")
        self.assertEqual(v["previous_84_legal_TRAIN_SHA256"],
            "a66833e8f4e83e4ba080dc3db0b3032a3315bcfc2fe7cee473e7e9ea7569023e")
        self.assertEqual(v["NEW_28_INTERNAL_DEV"],28)
        self.assertFalse(v["candidate_accepted"])
    def test_native_unmodified_baseline_is_full49152_greedy_only(self):
        original=torch.zeros((1,49152))
        original[0,890]=2.
        h=torch.zeros((1,320))
        embedding=torch.zeros((49152,320))
        chosen,meta=model.pick_second(original,h,first=7,embedding=embedding,ranker=None)
        self.assertEqual(chosen,890)
        self.assertEqual(meta["alpha"],0.)
        self.assertFalse(meta["changed"])
        self.assertFalse(original.requires_grad)
    def test_pairwise_score_requires_all_28_different_sources_and_three_arms(self):
        baseline=lambda i:[10+i,101+i,201+i]
        def make(i):
            o=baseline(i)
            first=i%4!=0
            result={}
            for label in model.MODES:
                tokens=o.copy()
                if label=="HUMAN_OWNFIRST_TRAIN_ONLY" and i<7:tokens[1]=42+i
                result[label]={
                    "native_first3_ids":tokens,
                    "correct_first":first,
                    "correct_first2":first and i<4 and label=="HUMAN_OWNFIRST_TRAIN_ONLY",
                    "correct_first3":first and i==1 and label=="HUMAN_OWNFIRST_TRAIN_ONLY",
                    "premature_EOS_first3":False,
                    "second":{"alpha":.5 if label!="ORIGINAL_NATIVE_GREEDY" else 0.,
                              "changed":label!="ORIGINAL_NATIVE_GREEDY"}
                }
            return {"source_id":f"Z46_{i}", "first_correct":first,
                    "original_second_gold_in_original_TOP64":True,
                    "results":result}
        entries=[make(i) for i in range(28)]
        score=model.scores(entries)
        self.assertEqual(score["modes"]["ORIGINAL_NATIVE_GREEDY"]["n"],28)
        self.assertEqual(score["modes"]["HUMAN_OWNFIRST_TRAIN_ONLY"]["correct_first3_native"],1)
        self.assertFalse(score["all_precommitted_nonpromoting_thresholds_pass"])
        self.assertFalse(score["candidate_accepted"])
        self.assertEqual(score["output_gate"],0)
        with self.assertRaises(ValueError):model.scores(entries[:-1])
        with self.assertRaises(ValueError):model.scores(entries[:27]+[entries[0]])
    def test_genuine_model_first_third_natively_greedy_on_own_BPE2(self):
        text=PHYSICAL.read_text(encoding="utf8")
        self.assertIn('row["prefix"]+tokens',text)
        self.assertIn('tokens.append(int(third[0].argmax().item()))',text)
        self.assertIn('first=int(first_logits[0].argmax().item())',text)
        self.assertIn('fit_all_old84(records,native_embedding',text)
        self.assertIn('ranker=fitted.get(mode)',text)
        self.assertIn('sha(z43_report)',text)
        self.assertIn('no_old_SPENT_DEV_teacher_forcing_optimizer_or_scoring',text)
        self.assertNotIn('for row in fresh_dev:',text)
    def test_no_previous_spent_dev_and_always_reject_deployment(self):
        v=source.verify()
        self.assertTrue(v["no_previous_SPENT_DEV_optimized_or_evaluated"])
        self.assertTrue(v["frozen_physical_config"]["DEV_optimizer_steps_zero"])
        self.assertTrue(v["frozen_physical_config"]["DEV_own_prefix_no_teacher_forcing"])
        self.assertEqual(v["output_gate"],0)
        self.assertEqual(model.MODES,(
            "ORIGINAL_NATIVE_GREEDY","HUMAN_OWNFIRST_TRAIN_ONLY",
            "WRONG17_OWNFIRST_TRAIN_CONTROL"))
if __name__=="__main__":
    unittest.main()
