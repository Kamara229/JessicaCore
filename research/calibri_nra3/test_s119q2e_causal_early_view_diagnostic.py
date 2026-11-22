"""S119q2e fail-closed causal mapping and source-only diagnostics."""
import unittest
from collections import Counter
import torch
from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q1_real43m_onepass_causal_early import (
    build_early_causal_view,PLACEHOLDER_MAPPING,
    single_pass_early_intervention,
)
from research.calibri_nra3.s119q2e_causal_early_view_diagnostic import (
    MIRROR_H0_H5,MAPS,EXPECTED_REAL_DECODER_CALLS,
    COMPONENT_PROMPTS,SOURCE_COUNT,trace_to_summary,
)
from research.calibri_nra3.s119p_real43m_onpath_component_ablation import (
    MASKS,temporary_gate_mask,
)


class FakeLayer(torch.nn.Module):
    def __init__(self,index):
        super().__init__()
        self.index=index
    def forward(self,x):
        return (x+float(self.index+1),)


class FakeCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model=torch.nn.Module()
        self.model.layers=torch.nn.ModuleList(
            FakeLayer(i) for i in range(12))
    def forward(self,x):
        for layer in self.model.layers:
            x=layer(x)[0]
        return x


class Adapter(torch.nn.Module):
    output_gate=0
    candidate_accepted=False
    activation_allowed=False
    def __init__(self):
        super().__init__()
        self.midlayer_gates=torch.nn.Parameter(
            torch.tensor([0.10,-0.06,0.03]))
        self.last=None
    def forward(self,x,*,contextual):
        self.last=x.detach().clone()
        factor=x[:,6,-1,:]-x[:,11,-1,:]
        return factor*torch.tanh(self.midlayer_gates[0])


def synthetic_trace():
    kinds=("roles_word_order","negation","transitive_comparison",
           "event_time","arithmetic","paraphrase")
    result=[]
    for k in kinds:
        for i in range(5):
            result.append({
                "kind":k,
                "source_id":k+str(i),
                "source_split":"CALIBRATION" if i<3 else "SOURCE_PROBE",
                "source_kind":"q2a" if i<3 else "q2c1",
                "mask":"trained_full","target_id":123,
                "native_target_CE":10.+i,
                "native_greedy_target":i==0,
                "H320_residual_norm":float(i),
            })
    return result


class S119Q2ECausalMappingContract(unittest.TestCase):
    def setUp(self):
        self.recorded={
            i:torch.full((1,5,320),float(i)) for i in range(6)
        }

    def test_01_mirror_is_first_six_early_original_states_twice(self):
        self.assertEqual(MIRROR_H0_H5,(0,1,2,3,4,5,0,1,2,3,4,5))
        self.assertEqual(MAPS["REPEAT_H5"],PLACEHOLDER_MAPPING)

    def test_02_mirror_extracts_only_real_early_original_layers(self):
        x=build_early_causal_view(self.recorded,mapping=MIRROR_H0_H5)
        self.assertEqual(tuple(x.shape),(1,12,5,320))
        for i in range(12):
            self.assertTrue(torch.equal(x[:,i],self.recorded[i%6]))

    def test_03_legacy_repeat_h5_view_is_exact_unchanged_default(self):
        x=build_early_causal_view(self.recorded)
        y=build_early_causal_view(
            self.recorded,mapping=PLACEHOLDER_MAPPING)
        self.assertTrue(torch.equal(x,y))
        for i in range(6,12):
            self.assertTrue(torch.equal(x[:,i],self.recorded[5]))

    def test_04_refuse_any_future_original_hidden_layer(self):
        mappings=[
            (0,1,2,3,4,5,6,1,2,3,4,5),
            (0,1,2,3,4,5,0,1,2,3,4,11),
            (0,1,2,3,4,5,0,1,2,3,4,-1),
            (0,1,2,3,4,5,0,1,2,3,4,5.0),
        ]
        for m in mappings:
            with self.subTest(mapping=m):
                with self.assertRaisesRegex(GateFailure,"CAUSALLY"):
                    build_early_causal_view(self.recorded,mapping=m)

    def test_05_refuse_mapping_that_changes_first_six_slots(self):
        mapping=(1,0,2,3,4,5,0,1,2,3,4,5)
        with self.assertRaisesRegex(GateFailure,"CAUSALLY"):
            build_early_causal_view(self.recorded,mapping=mapping)

    def test_06_refuse_missing_or_spoofed_actual_early_layer(self):
        missing=dict(self.recorded)
        missing.pop(4)
        with self.assertRaisesRegex(GateFailure,"exact early"):
            build_early_causal_view(missing,mapping=MIRROR_H0_H5)

    def test_07_one_actual_pass_yields_different_mirror_vs_legacy_hidden(self):
        model=FakeCore()
        child=Adapter()
        src=torch.ones(1,4,320)
        base=model(src).detach().clone()
        with single_pass_early_intervention(
            model,child,view_mapping=MIRROR_H0_H5) as w:
            changed=model(src)
        self.assertEqual(w["candidate_forward_calls"],1)
        self.assertEqual(w["placeholder_mapping"],list(MIRROR_H0_H5))
        self.assertNotEqual(float(w["adapter_delta_H320"]),0.)
        self.assertFalse(torch.equal(base,changed.detach()))

    def test_08_zero_gate_mirror_and_repeat_equal_original(self):
        model=FakeCore()
        child=Adapter()
        src=torch.ones(1,4,320)
        original=model(src)
        for mapping in MAPS.values():
            with temporary_gate_mask(child,MASKS["zero_exact_original"]):
                with single_pass_early_intervention(
                    model,child,view_mapping=mapping):
                    out=model(src)
            self.assertTrue(torch.equal(out.detach(),original.detach()))
        self.assertEqual(float(child.midlayer_gates[0]),0.1)

    def test_09_temporary_component_masks_restore_learned_weights(self):
        child=Adapter()
        old=child.midlayer_gates.detach().clone()
        for name in ("without_core","without_structure","without_planning"):
            with temporary_gate_mask(child,MASKS[name]):
                self.assertNotEqual(
                    child.midlayer_gates.detach().tolist(),old.tolist())
            self.assertTrue(torch.equal(child.midlayer_gates.detach(),old))

    def test_10_source_summary_fixed_balanced_and_not_blind(self):
        rows=synthetic_trace()
        r=trace_to_summary(rows)
        self.assertEqual(r["source_only_greedy_target_count"],6)
        self.assertEqual(r["mean_native_source_only_CE"],12.)
        self.assertEqual(r["mean_residual_H320_norm"],2.)
        self.assertTrue(r["read_only_same_author_templates_not_independent"])

    def test_11_reject_incomplete_real_source_diagnostic(self):
        with self.assertRaisesRegex(GateFailure,"same 30 old probes"):
            trace_to_summary(synthetic_trace()[:-1])

    def test_12_reject_train_rows_mislabeled_as_probe(self):
        rows=synthetic_trace()
        rows[0]["source_split"]="TRAIN"
        with self.assertRaisesRegex(GateFailure,"same 30 old probes"):
            trace_to_summary(rows)

    def test_13_reject_fake_native_vocab_target(self):
        rows=synthetic_trace()
        rows[0]["target_id"]=49152
        with self.assertRaisesRegex(GateFailure,"original full-vocab"):
            trace_to_summary(rows)

    def test_14_predeclared_physical_forward_budget(self):
        self.assertEqual(SOURCE_COUNT,30)
        self.assertEqual(COMPONENT_PROMPTS,12)
        self.assertEqual(EXPECTED_REAL_DECODER_CALLS,158)
        self.assertEqual(set(MAPS),{"REPEAT_H5","MIRROR_H0_H5"})


if __name__=="__main__":
    unittest.main()
