"""Z50 TRAIN-only causal marginal anti-collapse objective/regression."""
import unittest
from pathlib import Path
import torch
from research.calibri_nra3 import s119q2i8z50_trainonly_marginal_nativeBPE2_anticollapse as z50

P=Path(__file__).resolve().parent
PHYSICAL=P/"s119q2i8z50_physical43m_train84_marginal_ab.py"

class TestZ50MarginalTrainOnly(unittest.TestCase):
    def test_exact_precommitted_config(self):
        self.assertEqual((z50.EPOCHS,z50.SEED,z50.LR,z50.WEIGHT_DECAY),
                         (8,119145,.004,.04))
        self.assertEqual((z50.NATIVE_VOCAB,z50.TOP_K),(49152,64))
        self.assertEqual(len(z50.DOMAINS),7)
        self.assertEqual(z50.MARGINAL_KL_WEIGHT,.8)
        self.assertEqual(z50.MARGINAL_ENTROPY_DEFICIT_WEIGHT,.8)
        self.assertEqual(z50.MARGINAL_ENTROPY_TOLERANCE_BITS,.10)
        self.assertEqual(z50.MIN_CROSSPROMPT_ENTROPY_CHANGE_BITS,-.25)
    def test_hard_forward_selection_has_smooth_nonzero_gradient(self):
        self.assertEqual((z50.HARD_MASS,z50.SURROGATE_TEMPERATURE),(.95,.25))
        logits=torch.zeros(64,requires_grad=True)
        with torch.no_grad():
            logits[3]=.1
        p=z50.hard_marginal_probs(logits)
        self.assertEqual(int(p.argmax()),3)
        self.assertGreater(float(p[3].detach()),.95)
        self.assertAlmostEqual(float(p.detach().sum()),1.,places=5)
        (-p[3]).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()),0.)
    def test_hard_marginal_distinguishes_near_tie_collapse(self):
        ids=torch.arange(64)
        a=torch.zeros(64); b=torch.zeros(64)
        a[1]=.01; b[2]=.01
        c=torch.zeros((2,64),requires_grad=True)
        with torch.no_grad():
            c[:,3]=.01
        orig=[(ids,z50.hard_marginal_probs(v).detach()) for v in (a,b)]
        changed=[(ids,z50.hard_marginal_probs(v)) for v in c]
        loss,meta=z50.native_marginal_penalty(orig,changed)
        self.assertGreater(float(loss.detach()),.1)
        self.assertGreater(meta["original_train_marginal_entropy_bits"],
                           meta["shadow_train_marginal_entropy_bits"])
        loss.backward()
        self.assertTrue(torch.isfinite(c.grad).all())
        self.assertGreater(float(c.grad.abs().sum()),0.)
    def test_identity_marginal_has_no_penalty_despite_finite_grads(self):
        ids=torch.arange(64)
        p=torch.ones(64,dtype=torch.float32)/64
        learned=torch.zeros((2,64),requires_grad=True)
        q=torch.softmax(learned,dim=1)
        loss,meta=z50.native_marginal_penalty(
            [(ids,p.detach()),(ids,p.detach())],
            [(ids,q[0]),(ids,q[1])])
        self.assertLess(abs(float(loss.detach())),1e-6)
        self.assertLess(abs(meta["original_train_marginal_KL_to_shadow"]),1e-6)
        loss.backward()
        self.assertTrue(torch.isfinite(learned.grad).all())
    def test_crossprompt_collapse_penalized_even_with_per_prompt_distribution(self):
        # Original reference mass differs across TWO previous TRAIN prompts.
        ids=torch.arange(64)
        a=torch.full((64,),-8.)
        b=torch.full((64,),-8.)
        a[1]=4.
        b[2]=4.
        p1=torch.softmax(a,dim=0).detach()
        p2=torch.softmax(b,dim=0).detach()
        q_logits=torch.zeros((2,64),requires_grad=True)
        # Same token3 emerges from unrelated examples -> marginal collapse.
        with torch.no_grad():
            q_logits[:,3]=8.
        q=torch.softmax(q_logits,dim=1)
        loss,meta=z50.native_marginal_penalty(
            [(ids,p1),(ids,p2)],[(ids,q[0]),(ids,q[1])])
        self.assertGreater(float(loss.detach()),.1)
        self.assertGreater(meta["original_train_marginal_entropy_bits"],
                           meta["shadow_train_marginal_entropy_bits"])
        loss.backward()
        self.assertTrue(torch.isfinite(q_logits.grad).all())
        self.assertGreater(float(q_logits.grad.abs().sum()),0.)
    def test_marginal_readout_rejects_dev_and_fake_candidate_injection(self):
        ids=torch.arange(64)
        p=torch.ones(64)/64
        q=torch.softmax(torch.zeros(64,requires_grad=True),dim=0)
        with self.assertRaises(ValueError):
            z50.native_marginal_penalty([(ids,p)],[(ids,q)])
        with self.assertRaises(ValueError):
            z50.native_marginal_penalty([(ids,p),(ids,p)],[
                (ids,q),(ids.roll(1),q)])
        with self.assertRaises(ValueError):
            z50.native_marginal_penalty([(ids,p),(ids,p)],[
                (ids,torch.ones(64)/64),(ids,q)])
    def test_physical_previous84_only_original43m_model_unchanged(self):
        src=PHYSICAL.read_text(encoding="utf8")
        self.assertIn("original_forwards==168",src)
        self.assertIn("validate_onpolicy(records,native_embedding)",src)
        self.assertIn("old_original_param_digest(model)==original_digest",src)
        self.assertIn('comparisons=cross_domain(records,native_embedding)',src)
        self.assertIn('"Z46_NEW28_internal_DEV_is_SPENT_never_loaded":True',src)
        self.assertNotIn('for row in fresh_dev:',src)
        self.assertNotIn('fresh_new_rows("DEV")',src)
    def test_shadow_train_only_never_unfreeze_s142_or_gt12(self):
        src=Path(z50.__file__).read_text(encoding="utf8")
        self.assertIn('train=[r for r in records if r["kind"]!=heldout]',src)
        self.assertIn('loss.backward()',src)
        self.assertIn('one_group_marginal_loss(net,legal,embedding)',src)
        self.assertIn('"heldout_kind_shadow_optimizer_steps":0',src)
        self.assertIn('"old_SPENT_Z46_DEV28_never_fitted_or_rescored":True',src)
        self.assertIn('"candidate_accepted":False,"output_gate":0',src)
if __name__=="__main__":
    unittest.main()
