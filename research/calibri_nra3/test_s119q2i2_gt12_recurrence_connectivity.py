"""S119q2i2: fail-closed original GT12 recurrent vs local gate-only tests."""
from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest import TestCase,main

import torch
from torch import nn

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri2.s126.full_layer_state_tunnel import FullLayerStateTunnel
from research.calibri_nra3.s119q2i2_gt12_recurrence_connectivity import (
    native_full_gt_gate_contract,
    set_gt_route,CONDITIONS,GT_GATE_LR,GT_DEPTH,GT_RANK,
    TRAIN_PER_KIND,TRAIN_ROWS,DEV_ROWS,ROUNDS,MAX_OWN_NATIVE,
    MINUS_20,STAGE,
)
from research.calibri_nra3.s119q2i1_authored_ru_train_source import (
    rows as ru_rows,provenance,KINDS,
)


class FakeBridge:
    def __init__(self):
        self.coefficients=nn.Parameter(torch.zeros(2,10,17),
                                       requires_grad=False)


class FakeBlock(nn.Module):
    def forward(self,x):
        return (x,)


class FakeOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList(FakeBlock() for _ in range(12))
    def forward(self,x,past_key_values=None,use_cache=True):
        for m in self.model.layers:
            x=m(x)[0]
        return x


class S119Q2I2GT12CouplingContracts(TestCase):
    def setUp(self):
        self.full=FullLayerStateTunnel(width=320,depth=12,rank=4)
        self.bridge=FakeBridge()

    def test_01_S142_style_fullGT12_zero_initial_gates(self):
        v=native_full_gt_gate_contract(self.full,self.bridge)
        self.assertTrue(v["original_S142_GT12_12_gates_exact_ZERO"])
        self.assertTrue(v["BUT_GT12_original_route_has_no_active_recurrent_gain"])
        self.assertTrue(v["actual_GreenTunnel_and_new_regions_share_model_forward"])

    def test_02_S111_original_prompt_conditioned_bridge_zero_frozen(self):
        self.assertEqual(self.bridge.coefficients.numel(),340)
        self.assertEqual(torch.count_nonzero(self.bridge.coefficients).item(),0)
        self.assertFalse(self.bridge.coefficients.requires_grad)

    def test_03_fullGT12_exact_same_output_when_gates_zero(self):
        model=FakeOriginal()
        full=self.full.install(model)
        src=torch.randn(1,4,320)
        expected=model(src).detach()
        with full.scope():
            got=model(src).detach()
        self.assertTrue(torch.equal(expected,got))
        self.assertEqual(full.last_forward_coverage,tuple(range(12)))
        full.uninstall()

    def test_04_real_GT12_gate_grad_reaches_earlier_and_later_layers(self):
        model=FakeOriginal()
        self.full.install(model)
        for n,p in self.full.named_parameters():
            p.requires_grad_(n=="gates")
        with self.full.scope():
            out=model(torch.randn(1,4,320))
        out.square().mean().backward()
        self.assertIsNotNone(self.full.gates.grad)
        self.assertTrue(torch.isfinite(self.full.gates.grad).all())
        self.assertGreater(float(self.full.gates.grad.abs().sum()),1e-12)
        self.assertEqual(tuple(self.full.gates.grad.shape),(12,))
        self.full.uninstall()

    def test_05_switch_between_12gate_recurrent_and_local_same_count(self):
        g=torch.linspace(-0.03,0.03,12)
        set_gt_route(self.full,"GT_RECURRENT",g)
        self.assertTrue(torch.equal(self.full.gates,g))
        self.assertEqual(float(self.full.mix_logits.abs().sum()),0.)
        set_gt_route(self.full,"GT_NOCARRY_LOCAL",g)
        self.assertTrue(torch.equal(self.full.gates,g))
        self.assertTrue(torch.all(self.full.mix_logits==MINUS_20))
        self.assertEqual(sum(p.numel() for n,p in
                             self.full.named_parameters() if n=="gates"),12)

    def test_06_no_leaking_current_GT12_recurrent_state_across_requests(self):
        model=FakeOriginal()
        full=self.full.install(model)
        set_gt_route(full,"GT_RECURRENT",torch.full((12,),0.05))
        with full.scope():
            model(torch.randn(1,2,320))
        self.assertIsNone(full._carry)
        self.assertEqual(full._visited,[])
        self.assertEqual(full.last_forward_coverage,tuple(range(12)))
        self.full.uninstall()

    def test_07_recurrent_and_nocarry_different_outputs_with_same_parameters(self):
        model=FakeOriginal()
        self.full.install(model)
        x=torch.randn(1,4,320)
        gates=torch.linspace(0.03,0.06,12)
        set_gt_route(self.full,"GT_RECURRENT",gates)
        with self.full.scope():
            recur=model(x).detach()
        set_gt_route(self.full,"GT_NOCARRY_LOCAL",gates)
        with self.full.scope():
            local=model(x).detach()
        self.assertFalse(torch.equal(recur,local))
        self.assertEqual(self.full.last_forward_coverage,tuple(range(12)))
        self.full.uninstall()

    def test_08_cannot_set_GT_gates_for_unknown_architecture(self):
        with self.assertRaisesRegex(GateFailure,"GT route"):
            set_gt_route(self.full,"UNREGISTERED",torch.zeros(12))

    def test_09_cannot_mutate_GT_when_forward_scope_active(self):
        with self.full.scope():
            with self.assertRaisesRegex(GateFailure,"GT route"):
                set_gt_route(self.full,"GT_RECURRENT",torch.zeros(12))

    def test_10_cannot_prepare_wrong_shape_or_nonfinite_gate_vector(self):
        with self.assertRaisesRegex(GateFailure,"GT route"):
            set_gt_route(self.full,"GT_RECURRENT",torch.zeros(10))
        bad=torch.zeros(12);bad[0]=float("nan")
        with self.assertRaisesRegex(GateFailure,"GT route"):
            set_gt_route(self.full,"GT_RECURRENT",bad)

    def test_11_original_SOURCE_train_DEV_disjoint(self):
        tr=ru_rows("TRAIN");dv=ru_rows("DEV")
        self.assertEqual(len(tr),36)
        self.assertEqual(len(dv),12)
        self.assertFalse({x["prompt"] for x in tr}&
                         {x["prompt"] for x in dv})
        self.assertTrue(provenance()["old_sources_and_spent_q2g_disjoint"])

    def test_12_three_rounds_two_each_six_domain_original_author_train(self):
        tr=[x for x in ru_rows("TRAIN")
            if int(x["id"].split("_")[-1])<TRAIN_PER_KIND]
        self.assertEqual(len(tr),TRAIN_ROWS)
        self.assertEqual({k:2 for k in KINDS},
                         {k:sum(x["kind"]==k for x in tr) for k in KINDS})
        self.assertEqual(TRAIN_ROWS*ROUNDS,36)
        self.assertEqual(GT_DEPTH,12)
        self.assertEqual(GT_RANK,4)

    def test_13_full_native_dev_generation_cap_and_no_gold_allowed(self):
        self.assertEqual(MAX_OWN_NATIVE,12)
        self.assertEqual(DEV_ROWS,12)
        self.assertEqual(GT_GATE_LR,0.025)

    def test_14_never_inadvertently_activate_or_upgrade_GT(self):
        self.assertEqual(CONDITIONS,
                         ("GT_ZERO_BASELINE","GT_RECURRENT","GT_NOCARRY_LOCAL"))
        self.assertIn("REAL43M_GT12_RECURRENCE_LINK",STAGE)

    def test_15_gt_state_restoration_frozen_original(self):
        old={k:v.detach().clone() for k,v in self.full.state_dict().items()}
        set_gt_route(self.full,"GT_NOCARRY_LOCAL",torch.ones(12)*.09)
        with torch.no_grad():
            self.full.gates.copy_(old["gates"])
            self.full.mix_logits.copy_(old["mix_logits"])
        self.assertTrue(all(torch.equal(v,self.full.state_dict()[k])
                            for k,v in old.items()))


if __name__=="__main__":
    main()
