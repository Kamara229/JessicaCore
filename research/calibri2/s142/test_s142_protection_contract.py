"""S142 preflight: static protected TRAIN policy and CPU J7 invariants."""
from pathlib import Path
import ast
import unittest


class S142FirstProtectionContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=Path(__file__).with_name("train_one_j7_dual_cone.py")
        cls.source=cls.path.read_text(encoding="utf-8")
        cls.tree=ast.parse(cls.source)

    def test_precommit_correct_first_only_no_QA(self):
        p=self.source
        self.assertIn("for k in sorted(parent_correct_new_first)",p)
        self.assertIn('if new_routes[k]=="S93"',p)
        self.assertIn("protected_s122_first_token_margin(",p)
        self.assertIn("from calibri2.s142.dual_cone_adam import correct_adam_stage2_displacement",p)
        self.assertIn("g.detach()[:,:,448:472].clone()",p)
        self.assertIn("protected_new_first_loss_ids",p)
        self.assertNotIn("load_heldout",p)
        self.assertNotIn("from calibri2.s118",p)

    def test_single_bounded_real_transaction_and_retention(self):
        p=self.source
        for token in (
            "STEPS=2",
            "TRANSACTION_STEPS=2",
            "correct_adam_stage2_displacement(",
            "atomic_train_block(",
            "persisted_cpu_guard",
            "strict_train_retention(",
            "candidate_actual_CPU(",
            "S142_ACTUAL_43M_ONE_J7_FP16.safetensors",
            "S142_ONE_J7_ACTUAL_PHYSICAL_GUARD.json",
            "S142_REAL_43M_S122_FIRST_ANCHORS_J7_VERIFIED",
            "S142 committed J7 transaction without new TRAIN gain",
            'bpp>.7 or actual_speed["ratio"]<.9',
        ):
            with self.subTest(token=token):
                self.assertIn(token,p)
        self.assertNotIn("next_rank=authorize_growth(",p)
        self.assertIn('no_S136_growth_or_Jessica_promotion',p)

    def test_project_only_new_stage2_channels_but_keep_parent_exact(self):
        p=self.source
        self.assertIn("old_gradients.append(g.detach()[:,:,448:472].clone())",p)
        self.assertIn("novel_grad=expansion.up.grad.detach()[:,:,448:472].clone()",p)
        self.assertIn("expansion.up.grad=torch.zeros_like(expansion.up)",p)
        self.assertIn("expansion.up.grad[:,:,448:472].copy_(protected_direction)",p)
        self.assertIn("mask_gradient_to_new_channels_only(expansion)",p)
        self.assertIn("verify_frozen_inherited(expansion,immutable_before)",p)
        self.assertIn("final_old!=inherited_before or final_new!=inherited_new",p)


if __name__=="__main__":
    unittest.main()
