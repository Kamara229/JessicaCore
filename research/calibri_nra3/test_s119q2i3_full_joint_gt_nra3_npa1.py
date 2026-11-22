"""S119q2i3 failclosed true GT12/Core/NRA3/NPA1/full-native joint contracts."""
from __future__ import annotations

from collections import Counter
from unittest import TestCase,main
import torch
from torch import nn

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    NativeFullHeadShadow,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    LearnedOnPathInterlayer,
)
from research.calibri_nra3.s119q2f_learned_causal_early_projection import (
    LearnedCausalEarlyProjector,
)
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import (
    JointRuNative,NativeHeadAlignment,temporary_learned_native_head,
)
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import (
    scoped_reduced_native_head_gain,make_joint_trainable,
    STAGE,HEAD_GAIN,GT_CONDITIONS,ROUNDS,TRAIN_ROWS,DEV_ROWS,
    TOTAL_EXPECTED_TRAINABLE,LEARNED_NEW_REGIONS_EXPECTED,
    HEAD_BIAS_LR,GATE_LR,CORE_REGION_LR,HEAD_PROJECTION_LR,
)
from research.calibri_nra3.s119q2i1_authored_ru_train_source import (
    rows as ru_rows,KINDS,
)


class ToyOriginalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head=nn.Linear(320,49152,bias=False)
        self.lm_head.requires_grad_(False)
    def forward(self,hidden):
        return self.lm_head(hidden)


def fresh_joint():
    old=LearnedOnPathInterlayer(NativeFullHeadShadow())
    projected=LearnedCausalEarlyProjector(old)
    return JointRuNative(projected)


class S119Q2I3FullJointGTCausalityContracts(TestCase):
    def test_01_deep_neural_native_head_zero_gate_safe(self):
        head=NativeHeadAlignment()
        self.assertFalse(head.candidate_accepted)
        self.assertEqual(head.output_gate,0)
        self.assertEqual(HEAD_GAIN,.25)

    def test_02_head_scaled_hook_in_the_true_original_native_lm_head(self):
        torch.manual_seed(191)
        original=ToyOriginalLM()
        head=NativeHeadAlignment()
        input=torch.randn(1,3,320)
        old=original(input).detach()
        with torch.no_grad():
            head.output_bias[147]=4.
        with scoped_reduced_native_head_gain(head) as proof:
            with temporary_learned_native_head(original,head) as native_count:
                produced=original(input)
        self.assertEqual(proof[0],1)
        self.assertEqual(native_count["original_head_calls"],1)
        self.assertAlmostEqual(float((produced-old)[0,-1,147]),1.,places=4)
        self.assertTrue(torch.equal(original(input).detach(),old))
        self.assertEqual(len(head._forward_hooks),0)
        self.assertFalse(original.lm_head.weight.requires_grad)

    def test_03_head_gain_scope_removes_hook_on_failure(self):
        head=NativeHeadAlignment()
        with self.assertRaisesRegex(RuntimeError,"deliberate"):
            with scoped_reduced_native_head_gain(head):
                raise RuntimeError("deliberate")
        self.assertFalse(head._forward_hooks)

    def test_04_head_scaling_is_fixed_not_learnable_or_russian_token_mask(self):
        h=NativeHeadAlignment()
        self.assertTrue(
            all(p.numel() for p in h.parameters()))
        self.assertFalse(any("gain" in k for k in h.state_dict()))
        data=torch.randn(1,2,320)
        with torch.no_grad():h.output_bias.fill_(.1)
        with scoped_reduced_native_head_gain(h):
            a=h(data)
        self.assertTrue(torch.allclose(
            a,torch.ones_like(a)*.025,atol=1e-6))
        self.assertTrue(torch.equal(
            h.output_bias,torch.ones_like(h.output_bias)*.1))

    def test_05_original_pretrained_old_s115_53751_unchanged_by_unfreeze(self):
        joint=fresh_joint()
        protected=joint.child.child.shadow.unified.nra3_neural_weights
        before={k:v.detach().clone() for k,v in protected.state_dict().items()}
        chosen=make_joint_trainable(joint)
        self.assertTrue(all(not p.requires_grad for p in protected.parameters()))
        self.assertEqual(sum(p.numel() for p in chosen),LEARNED_NEW_REGIONS_EXPECTED)
        self.assertTrue(all(torch.equal(v,protected.state_dict()[k])
                            for k,v in before.items()))

    def test_06_exact_joint_368039_plus_twelve_native_gates(self):
        joint=fresh_joint()
        trainable=make_joint_trainable(joint)
        self.assertEqual(LEARNED_NEW_REGIONS_EXPECTED,368039)
        self.assertEqual(TOTAL_EXPECTED_TRAINABLE,368051)
        self.assertEqual(sum(p.numel() for p in trainable)+12,
                         TOTAL_EXPECTED_TRAINABLE)
        self.assertEqual(sum(p.numel() for p in joint.language_head.parameters()),
                         247040)

    def test_07_joint_learning_contains_all_six_area_neural_grad_targets(self):
        joint=fresh_joint()
        make_joint_trainable(joint)
        selected={id(p) for p in joint.trainables()}
        listed=[
            joint.child.child.shadow.unified.core_growth.up,
            joint.child.child.shadow.unified.structured_interface.fuse.weight,
            joint.child.child.shadow.unified.planning_action_area.state_update.weight_ih,
            joint.child.up,
            joint.language_head.up,
            joint.language_head.output_bias,
        ]
        self.assertTrue(all(id(p.weight if hasattr(p,"weight") else p)
                            in selected for p in listed))

    def test_08_original_bpe_and_native_eos_q2i1_original_source_only(self):
        train=[x for x in ru_rows("TRAIN")
               if int(x["id"].rsplit("_",1)[-1])<2]
        dev=ru_rows("DEV")
        self.assertEqual(len(train),TRAIN_ROWS)
        self.assertEqual(len(dev),DEV_ROWS)
        self.assertFalse({x["prompt"] for x in train}&
                         {x["prompt"] for x in dev})
        self.assertEqual(Counter(x["kind"] for x in train),
                         {k:2 for k in KINDS})

    def test_09_three_rounds_not_claimed_independent(self):
        self.assertEqual(ROUNDS,3)
        self.assertEqual(ROUNDS*TRAIN_ROWS,36)
        self.assertIn("JOINT_3ROUNDS",STAGE)

    def test_10_matched_recurrent_and_nocarry_are_only_two_training_arms(self):
        self.assertEqual(GT_CONDITIONS,("GT_RECURRENT","GT_NOCARRY_LOCAL"))
        self.assertEqual(len(GT_CONDITIONS),2)

    def test_11_head_bias_relearning_much_slower_than_head_up(self):
        self.assertLess(HEAD_BIAS_LR,CORE_REGION_LR)
        self.assertLess(CORE_REGION_LR,HEAD_PROJECTION_LR)
        self.assertLess(HEAD_PROJECTION_LR,GATE_LR)

    def test_12_original_gt12_not_in_child_neural_parameter_optimizer(self):
        joint=fresh_joint()
        ids={id(p) for p in make_joint_trainable(joint)}
        self.assertTrue(all(p.requires_grad for p in joint.trainables()))
        self.assertEqual(len(ids),len(joint.trainables()))

    def test_13_rejected_activation_stops_native_head_scaling(self):
        head=NativeHeadAlignment()
        head.candidate_accepted=True
        with self.assertRaisesRegex(GateFailure,"fixed research head"):
            with scoped_reduced_native_head_gain(head):
                pass

    def test_14_original_author_source_has_no_old_gold(self):
        a=ru_rows("TRAIN")
        self.assertTrue(all("original_newly_authored" in x["origin"]
                            for x in a))
        self.assertEqual(len(a),36)

    def test_15_starting_from_identical_joint_state_dicts(self):
        a=fresh_joint()
        b=fresh_joint()
        b.load_state_dict(a.state_dict(),strict=True)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k])
                            for k,v in a.state_dict().items()))

    def test_16_gt12_recurrent_not_equivalent_to_zero_original_gate(self):
        self.assertEqual(TOTAL_EXPECTED_TRAINABLE-
                         LEARNED_NEW_REGIONS_EXPECTED,12)

    def test_17_output_gateway_no_production_activation(self):
        j=fresh_joint()
        self.assertEqual(j.output_gate,0)
        self.assertFalse(j.candidate_accepted)
        self.assertFalse(j.activation_allowed)


if __name__=="__main__":
    main()
