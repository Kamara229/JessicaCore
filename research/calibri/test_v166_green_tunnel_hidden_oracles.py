"""Synthetic safe tests: read-only teacher oracles are NOT a trained tunnel."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn

from v166.protocol import (check,TEACHER_ONLY_STAGES,FIT_DOMAINS,ARM_NAMES,
                           WINDOW,TOKENS_PER_DOMAIN,PAID_V161_BYTES)
from v166.hooks import (stage_module,extract_hidden,record_reference,
                        inject_illegal_teacher_oracle)
from v166.evidence import check_v165
from v166.audit import summarize


class FakeOriginalLayers(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.embed_tokens=nn.Embedding(64,576)
        self.model.layers=nn.ModuleList([nn.Identity() for _ in range(30)])
        self.model.norm=nn.Identity()

    def forward(self,ids):
        hidden=self.model.embed_tokens(ids)
        for layer in self.model.layers:hidden=layer(hidden)
        return self.model.norm(hidden)


def row(nll,kl,top1=0):
    return {
        "tokens_scored":63,
        "sum_NLL":63*nll,
        "sum_teacher_NLL":63*4.1,
        "sum_teacher_to_student_KL":63*kl,
        "exact_FP32_next_token_argmax_count":top1,
    }


class GreenTunnelCausalOracleTests(unittest.TestCase):
    def test_frozen_stages_and_fitted_old_source_only(self):
        self.assertTrue(check())
        self.assertEqual(TEACHER_ONLY_STAGES,
            ("input_embedding","block_00_output","block_09_output",
             "block_28_output","final_norm"))
        self.assertEqual((WINDOW,TOKENS_PER_DOMAIN),(64,252))
        self.assertEqual(FIT_DOMAINS,("wiki_fit","news_fit"))

    def test_real_model_hooks_replace_hidden_and_do_not_leak(self):
        net=FakeOriginalLayers()
        tokens=torch.arange(WINDOW)[None,:]%64
        with record_reference(net) as hidden:
            original=net(tokens)
        self.assertEqual(tuple(hidden),TEACHER_ONLY_STAGES)
        self.assertTrue(all(x.shape==(1,64,576) for x in hidden.values()))
        arbitrary=torch.ones_like(original)*123
        with inject_illegal_teacher_oracle(net,"block_09_output",arbitrary):
            result=net(tokens)
        self.assertTrue(torch.all(result==123))
        self.assertEqual(len(net.model.layers[9]._forward_hooks),0)
        self.assertFalse(torch.all(net(tokens)==123))
        with self.assertRaises(ValueError):
            with inject_illegal_teacher_oracle(net,"block_01_output",arbitrary):
                pass
        with self.assertRaisesRegex(ValueError,"artificial"):
            with record_reference(net):
                raise ValueError("artificial")
        self.assertEqual(len(net.model.embed_tokens._forward_hooks),0)
        ref=torch.zeros((1,64,576))
        self.assertEqual(extract_hidden((ref,"mock_KV")).shape,ref.shape)
        with self.assertRaises(ValueError):
            extract_hidden(torch.zeros((1,64,575)))

    def test_real_v165_top128_failure_must_be_documented_not_guessed(self):
        original={
            "stage":"v165_NO_TRAINING_GREEN_TUNNEL_TOPK_RESCUE_UPPER_BOUND",
            "actual_frozen_CB161_sha256":"1"*64,
            "no_new_weights_or_external_tunnel_trained":True,
            "GREEN_TUNNEL_TOPK_oracle_upper_bounds_not_measured_Green_Tunnel":{
                "fully_paid_layer0_CB161":{
                    d:{
                        "topK_support_information_theoretic_reranker_ceiling_not_performance":{
                            "128":{"teacher_TOP1_recovery_upper_bound_fraction":small},
                            "2048":{"teacher_TOP1_recovery_upper_bound_fraction":large},
                        }
                    }
                    for d,small,large in (
                        ("wiki_fit",.115,.607),
                        ("news_fit",.095,.373))
                }
            },
        }
        physical=SimpleNamespace(sha256="1"*64,size=PAID_V161_BYTES,
                                 layer=0,patches={str(i):True for i in range(7)})
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"v165.json"
            p.write_text(json.dumps(original))
            self.assertTrue(check_v165(p,physical)[
                "reason_to_test_earlier_hidden_activation_points"])
            original["GREEN_TUNNEL_TOPK_oracle_upper_bounds_not_measured_Green_Tunnel"][
                "fully_paid_layer0_CB161"]["news_fit"][
                    "topK_support_information_theoretic_reranker_ceiling_not_performance"][
                        "128"]["teacher_TOP1_recovery_upper_bound_fraction"]=.99
            p.write_text(json.dumps(original))
            with self.assertRaises(ValueError):
                check_v165(p,physical)

    def test_matched_five_teacher_oracles_can_fail_without_claiming_GT(self):
        records={
            arm:{d:{
                "baseline":[row(15.,11.,0) for _ in range(4)],
                **{stage:[row(14.,10.,2) for _ in range(4)]
                   for stage in TEACHER_ONLY_STAGES},
            } for d in FIT_DOMAINS}
            for arm in ARM_NAMES
        }
        measured=summarize(records)
        self.assertTrue(measured[
            "theoretical_teacher_activation_transplant_NOT_a_trained_10MB_Green_Tunnel"])
        self.assertAlmostEqual(measured["actual_old_TRAIN_FIT_source_only"][
            "original_CB59"]["wiki_fit"]["five_oracle_hidden_state_boundary_effects"][
            "block_00_output"][
            "UNPAID_activation_ORACLE_NLL_delta_vs_original_packed"],-1.)
        records["paid_original_block0"]["news_fit"]["final_norm"].pop()
        with self.assertRaises(ValueError):
            summarize(records)


if __name__=="__main__":
    unittest.main()
