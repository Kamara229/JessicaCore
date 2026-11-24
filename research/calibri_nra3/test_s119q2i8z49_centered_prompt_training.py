"""Z49 stdlib/torch small source contract; no new original43M physical scoring."""
import unittest
import torch
from pathlib import Path
from research.calibri_nra3 import s119q2i8z49_trainonly_prompt_centered_native_bpe2 as z49

class TestZ49CenteredPromptShadow(unittest.TestCase):
    def test_architectural_one_change_and_per_domain_training_budget(self):
        self.assertEqual((z49.EPOCHS,z49.LR,z49.WEIGHT_DECAY),(8,.004,.04))
        self.assertEqual((len(z49.FOLDS),z49.SEED),(7,119145))
        self.assertEqual(z49.MIN_ELIGIBLE,31)
        self.assertEqual(z49.MIN_CORRECT_FIRST2,6)
        self.assertEqual(z49.MIN_DISTINCT_FRACTION,.75)
        self.assertEqual(z49.MIN_EMPIRICAL_ENTROPY_DELTA_BITS,-.25)
        self.assertFalse(z49.SHADOW_ACCEPTED)
        self.assertEqual(z49.OUTPUT_GATE,0)
        self.assertTrue(z49.NO_DEV)
    def test_training_center_uses_only_72_detached_contexts(self):
        samples=[{"split":"TRAIN",
                  "h320":torch.full((1,320),float(i%5)),
                  "prev_own_embed":torch.arange(320,dtype=torch.float32).view(1,-1)}
                 for i in range(72)]
        center=z49.paired_context(samples)
        self.assertEqual(tuple(center.shape),(1,640))
        self.assertFalse(center.requires_grad)
        self.assertTrue(torch.isfinite(center).all())
        with self.assertRaises(ValueError):z49.paired_context(samples[:71])
        with self.assertRaises(ValueError):z49.paired_context(
            samples[:71]+[dict(samples[71],split="DEV")])
    def test_new_native_ranker_retains_original_targetblind64_and_finite_grad(self):
        torch.manual_seed(42)
        center=torch.randn((1,640))
        model=z49.ContextCenteredNativeBPE2Comparator(center)
        self.assertEqual(sum(p.numel() for p in model.parameters()),30720)
        self.assertFalse(model.train_context_mean.requires_grad)
        self.assertEqual(model.output_gate,0)
        self.assertFalse(model.candidate_accepted)
        orig=torch.zeros((1,49152))
        orig[0,234]=3.
        h=torch.randn((1,320))
        prev=torch.randn((1,320))
        embedding=torch.randn((49152,320))
        indices,score,resid=model(orig,h,prev,embedding,native_bpe_position=2)
        self.assertEqual((indices.numel(),indices.unique().numel()),(64,64))
        self.assertTrue((indices==234).any())
        self.assertTrue(torch.isfinite(score).all())
        self.assertLessEqual(float(resid.abs().max()),1.50001)
        ce=torch.nn.functional.cross_entropy(
            score.view(1,-1),torch.tensor([0],dtype=torch.long))
        ce.backward()
        self.assertTrue(all(p.grad is not None and
                            torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        self.assertIsNone(orig.grad)
        self.assertIsNone(embedding.grad)
        with self.assertRaises(ValueError):
            model(orig,h,prev,embedding,native_bpe_position=3)
    def test_requires_frozen_context_input_no_gpu_or_dev(self):
        with self.assertRaises(ValueError):
            z49.ContextCenteredNativeBPE2Comparator(
                torch.randn(1,640,requires_grad=True))
        self.assertEqual(z49.STAGE,
            "S119Q2I8Z49_ORIGINAL43M_LEGAL_TRAIN84_ONLY_CENTERED_OWN_BPE2_LODO7")
        src=Path(z49.__file__).read_text(encoding="utf8")
        self.assertIn('r["kind"]!=heldout',src)
        self.assertIn('r["split"]=="TRAIN"',src)
        self.assertNotIn('split="DEV"',src)
        self.assertNotIn("z48.audit(",src)
        self.assertNotIn("z47.scores(",src)
        self.assertIn('"candidate_accepted":False,"output_gate":0',src)
if __name__=="__main__":
    unittest.main()
