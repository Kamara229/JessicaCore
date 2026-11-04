"""Non-TEST 30-block hooks, report semantics and prior FAIL validations."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from v159.protocol import (check_protocol,TRACE_KEYS,EXPECTED_STAGES,
                           LAYERS,FIT_DOMAINS,CB153_SHA)
from v159.trace import run_original_fit_trace,hook_all_layers
from v159.audit import by_layer,compare_domain,summarize
from v159.evidence import verify_prior


class FakeOriginalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.embed_tokens=nn.Embedding(50000,576)
        self.model.layers=nn.ModuleList([nn.Identity() for _ in range(LAYERS)])
        self.model.norm=nn.Identity()

    def forward(self,input_ids,use_cache=False):
        h=self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            h=layer(h)
        h=self.model.norm(h)
        # Avoid actual 49152 logits allocation in synthetic hook tests.
        return SimpleNamespace(logits=torch.zeros((1,64,49152)))


class F159Tests(unittest.TestCase):
    def test_source_30_layers_and_32_trace_points_are_locked(self):
        self.assertTrue(check_protocol())
        self.assertEqual(len(TRACE_KEYS),EXPECTED_STAGES)
        self.assertEqual(EXPECTED_STAGES,32)
        self.assertEqual(TRACE_KEYS[0],"input_embedding")
        self.assertEqual(TRACE_KEYS[7],"block_06_output")
        self.assertEqual(TRACE_KEYS[-1],"final_norm")
        self.assertEqual(FIT_DOMAINS,("wiki_fit","news_fit"))

    def test_full_actual_module_hooks_no_leak_and_matched_256_positions(self):
        fake=FakeOriginalModel()
        tokens=torch.arange(256,dtype=torch.long)%100
        source=run_original_fit_trace(fake,tokens)
        self.assertEqual(tuple(source),TRACE_KEYS)
        self.assertTrue(all(s.shape==(256,576) for s in source.values()))
        # Hook counts must not accumulate after leaving the context.
        self.assertEqual(len(fake.model.layers[0]._forward_hooks),0)
        self.assertEqual(len(fake.model.embed_tokens._forward_hooks),0)
        with self.assertRaises(ValueError):
            run_original_fit_trace(fake,tokens[:-1])
        with self.assertRaisesRegex(ValueError,"synthetic"):
            with hook_all_layers(fake):
                raise ValueError("synthetic")
        self.assertEqual(len(fake.model.layers[7]._forward_hooks),0)

    def test_activation_drift_by_depth_not_false_causal_attribution(self):
        original={stage:torch.ones((256,8),dtype=torch.float32)
                  for stage in TRACE_KEYS}
        ordinary={stage:value*.5 for stage,value in original.items()}
        paid={stage:(v*.5 if index<=7 else v*.75)
              for index,(stage,v) in enumerate(original.items())}
        result=compare_domain(original,ordinary,paid)
        self.assertAlmostEqual(
            result["input_tied_embedding_original_CB59_RMSE"],.5,places=6)
        self.assertAlmostEqual(
            result["paid_vproj_changes_drift_at_block6"],0.,places=6)
        self.assertLess(result["paid_vproj_changes_drift_after_final_norm"],0.)
        self.assertTrue(
            result["cannot_attribute_cause_to_one_layer_from_post_activation_RMSE_alone"])
        report=summarize({d:result for d in FIT_DOMAINS})
        self.assertEqual(len(report["common_heaviest_5_RMSE_increments"]),2)
        with self.assertRaises(ValueError):
            by_layer(original,dict(list(ordinary.items())[:-1]))

    def test_actual_prior_v157_FAIL_v158_upstream_dominance_required(self):
        v157={
          "stage":"v157_FOURTH_UNSEEN_SOURCE_32WINDOW_VPROJ_OUTLIER_ROBUSTNESS",
          "immutable_complete_CB153_SHA256":CB153_SHA,
          "window_level_outlier_and_strong_two_domain_gate":{
            "v_proj_STRONG_two_domain_robustness_pass":False,
            "honest_fourth_NEW_source_results":{
              "wiki_fourth_robustness_VALID":{
                "v_proj_minus_CB59_gold_NLL":.0372,
                "v_proj_minus_CB59_teacher_KL":.0404,
                "precommitted_STRONG_pass":False,
              },
              "news_fourth_robustness_VALID":{
                "precommitted_STRONG_pass":False,
              }
            }
          },
        }
        v158={
          "stage":"v158_ORIGINAL_LAYER6_VPROJ_WEIGHT_ERROR_VS_UPSTREAM_DRIFT",
          "exact_frozen_paid_CB153_sha256":CB153_SHA,
          "findings":{
            "original_TRAlN_FIT_only_domain_findings":{
               d:{
                 "upstream_drift_dominates_original_weight_only_error":True,
                 "paid_1bit_reduces_combined_actual_original_operator_error":False,
                 "ratio_upstream_drift_vs_original_5bit_weight_only_error":2.8
               } for d in FIT_DOMAINS,
            }
          }
        }
        with tempfile.TemporaryDirectory() as t:
            p,q=Path(t)/"v157.json",Path(t)/"v158.json"
            p.write_text(json.dumps(v157))
            q.write_text(json.dumps(v158))
            observed=verify_prior(p,q)
            self.assertEqual(observed["v157_real_32window_two_domain_strong_gate"],"FAIL")
            self.assertEqual(observed["v158_original_fit_upstream_drift_dominance_ratios"]["wiki_fit"],2.8)
            v157["window_level_outlier_and_strong_two_domain_gate"]["v_proj_STRONG_two_domain_robustness_pass"]=True
            p.write_text(json.dumps(v157))
            with self.assertRaises(ValueError):
                verify_prior(p,q)


if __name__=="__main__":
    unittest.main()
