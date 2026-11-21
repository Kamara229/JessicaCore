"""S119p small synthetic unit tests; no hidden evaluation sets / model weights."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119p_real43m_onpath_component_ablation import (
    COMPONENTS, PROMPTS, MASKS, first_token_metrics,
    temporary_gate_mask, verify_original_research_report,
)
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import sha
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import (
    STAGE as S119N_STAGE,
)


def sample():
    original = torch.tensor([[5., 1., 0., -1., -2.]])
    full = original + torch.tensor([[.1, .03, .0, .0, .0]])
    variants = {
        "trained_full": full,
        "without_core": original + torch.tensor([[.04, .02, 0., 0., 0.]]),
        "without_structure": original + torch.tensor([[.06, .01, 0., 0., 0.]]),
        "without_planning": original + torch.tensor([[.08, .00, 0., 0., 0.]]),
        "zero_exact_original": original.clone(),
    }
    components = torch.zeros(1,3,320)
    components[0,0,0] = 1.
    components[0,1,1] = 2.
    components[0,2,2] = 3.
    gates = torch.full((3,), 0.1)
    return original, variants, components, gates


class TinyAdapter:
    def __init__(self):
        self.midlayer_gates = torch.nn.Parameter(
            torch.tensor([.1,-.05,.007]))


class S119PResearchControls(unittest.TestCase):
    def test_01_actual_full_vocab_proxy_math_not_answer_quality(self):
        out = first_token_metrics(*sample())
        self.assertEqual(out["original_native_first_id"], 0)
        self.assertEqual(out["trained_full_native_first_id"], 0)
        self.assertTrue(out["same_native_first_token"])
        self.assertTrue(out["sufficient_argmax_stability_margin_gt_2x_Linf"])
        self.assertAlmostEqual(out["original_top1_top2_margin"], 4.)
        self.assertEqual(set(out["weighted_H320_component_norms"]), set(COMPONENTS))
        self.assertTrue(out["zero_gate_full_vocab_original_parity"])

    def test_02_components_can_cancel_even_with_nonzero_gates(self):
        base, variants, components, gates = sample()
        components.zero_()
        components[0,0,0] = 1.
        components[0,1,0] = -1.
        components[0,2,1] = 0.
        out = first_token_metrics(base, variants, components, gates)
        self.assertAlmostEqual(
            out["component_cancellation_ratio_0_none_1_total"], 1.,
            places=5)
        self.assertAlmostEqual(out["full_H320_residual_norm"], 0., places=5)

    def test_03_no_false_causal_claim_from_leave_one_out(self):
        out = first_token_metrics(*sample())
        for item in out["causal_component_leave_one_out_native_logits"].values():
            self.assertGreaterEqual(
                item["max_abs_full_vocab_logits_FULL_vs_without"], 0.)
            self.assertIs(item["candidate_token_changed_when_component_omitted"],
                          False)

    def test_04_reject_zero_gate_full_vocab_parity_failure(self):
        args = list(sample())
        args[1]["zero_exact_original"] = args[0] + 0.5
        with self.assertRaisesRegex(GateFailure, "ZERO-gate"):
            first_token_metrics(*args)

    def test_05_reject_nan_logits_in_any_predeclared_ablation(self):
        args = list(sample())
        args[1]["without_core"][0,2] = float("nan")
        with self.assertRaisesRegex(GateFailure, "finite"):
            first_token_metrics(*args)

    def test_06_reject_unregistered_gate_amplification(self):
        adapter = TinyAdapter()
        with self.assertRaisesRegex(GateFailure, "pre-registered"):
            with temporary_gate_mask(adapter, (4,1,1)):
                pass

    def test_07_temporary_mask_restores_gates_after_exception(self):
        adapter = TinyAdapter()
        prior = adapter.midlayer_gates.detach().clone()
        for mask in MASKS.values():
            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                with temporary_gate_mask(adapter, mask):
                    expected = prior * torch.tensor(mask)
                    self.assertTrue(torch.equal(adapter.midlayer_gates, expected))
                    raise RuntimeError("synthetic failure")
            self.assertTrue(torch.equal(adapter.midlayer_gates, prior))

    def test_08_reject_missing_immutable_ancestors(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            files = []
            for idx in range(3):
                fp = p / ("w"+str(idx))
                fp.write_bytes(("source-"+str(idx)).encode())
                files.append(fp)
            records = [{
                "category": group,
                "source_question": question,
                "initial_question_SHA256":
                    hashlib.sha256(question.encode("utf8")).hexdigest(),
                "answer_grade_NOT_MEASURED": True,
                "different_native_generated_token_sequences": False,
                "original_native_greedy": {
                    "generated_native_token_ids": [0]},
                "actual_deep_onpath_native_greedy": {
                    "generated_native_token_ids": [0]},
            } for group,question in PROMPTS]
            parent = {
                "stage":S119N_STAGE,"output_gate":0,
                "candidate_accepted":False,
                "physical_full_original_native_vocab_size":49152,
                "n_unlabeled_author_prompts":6,
                "prompts_where_trained_onpath_changes_generated_native_BPE":0,
                "total_actual_original43m_forward_calls":54,
                "no_heldout_QA_gold_answers":True,
                "original_616_J7_EOS_NOT_replayed":True,
                "no_live_activation":True,
                "original_S93_CB61_SHA256":sha(files[0]),
                "original_S142_43M_GT_SHA256":sha(files[1]),
                "S119m_trained_complete_physical_FP16_SHA256":sha(files[2]),
                "research_records":records,
            }
            self.assertEqual(verify_original_research_report(
                parent,*files),records)
            parent["original_S142_43M_GT_SHA256"]="0"*64
            with self.assertRaisesRegex(GateFailure,"ancestry"):
                verify_original_research_report(parent,*files)

    def test_09_reject_fake_graded_sources(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            files = []
            for idx in range(3):
                fp=p/str(idx)
                fp.write_bytes(str(idx).encode())
                files.append(fp)
            parent = {
                "stage":S119N_STAGE,"output_gate":0,
                "candidate_accepted":False,
                "physical_full_original_native_vocab_size":49152,
                "n_unlabeled_author_prompts":6,
                "prompts_where_trained_onpath_changes_generated_native_BPE":0,
                "total_actual_original43m_forward_calls":54,
                "no_heldout_QA_gold_answers":True,
                "original_616_J7_EOS_NOT_replayed":True,
                "no_live_activation":True,
                "original_S93_CB61_SHA256":sha(files[0]),
                "original_S142_43M_GT_SHA256":sha(files[1]),
                "S119m_trained_complete_physical_FP16_SHA256":sha(files[2]),
                "research_records":[{
                    "category":group,"source_question":question,
                    "initial_question_SHA256":hashlib.sha256(
                        question.encode("utf8")).hexdigest(),
                    "answer_grade_NOT_MEASURED":False,
                    "different_native_generated_token_sequences":False,
                    "original_native_greedy":{"generated_native_token_ids":[1]},
                    "actual_deep_onpath_native_greedy":{
                        "generated_native_token_ids":[1]},
                } for group, question in PROMPTS],
            }
            with self.assertRaisesRegex(GateFailure,"GOLD"):
                verify_original_research_report(parent,*files)

    def test_10_argmax_margin_small_is_not_a_failure_if_still_equal(self):
        base, variants, components, gates = sample()
        base[0,1] = 4.985
        variants["zero_exact_original"] = base.clone()
        variants["trained_full"] = base + torch.tensor(
            [[.01,-.01,0.,0.,0.]])
        out = first_token_metrics(base, variants, components, gates)
        self.assertTrue(out["same_native_first_token"])
        self.assertFalse(out["sufficient_argmax_stability_margin_gt_2x_Linf"])

    def test_11_dimension_change_or_missing_component_rejected(self):
        base, variants, components, gates = sample()
        del variants["without_planning"]
        with self.assertRaisesRegex(GateFailure,"three physical H320"):
            first_token_metrics(base,variants,components,gates)


if __name__ == "__main__":
    unittest.main()
