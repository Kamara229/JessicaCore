"""CALIBRI v92 deterministic safety and full-block encoding regressions, no TEST."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from direct_codebook_v59 import encode,CB59
from v92.archive import inplace_replace
from v92.encode import reencode_8d,weighted_distortion
from v92.policy import require_research_only,ApprovalRequired,decide_candidate,physical_budget
from v92.pipeline import priority_target
from v92.evidence import triage,ingest

class TestV92(unittest.TestCase):
    def test_entire_block_full_original_bytes_no_extra_bits_and_all_other_weights_identical(self):
        # Two physical learned tensor matrices, one target rewrites its codes+centroids.
        initial={"a.weight":torch.randn(32,16,generator=torch.Generator().manual_seed(92)),
                 "b.weight":torch.randn(16,16,generator=torch.Generator().manual_seed(93))}
        original,_,_=encode(initial)
        old=CB59(original)
        w=initial["a.weight"].numpy().copy()
        train=np.linspace(.3,2.,16)
        valid=np.linspace(.4,2.2,16)
        changed=reencode_8d(w,old.handles["a.weight"],train,valid,iterations=3)
        rebuilt,report=inplace_replace(original,"a.weight",changed)
        new=CB59(rebuilt)
        self.assertEqual(len(original),len(rebuilt))
        self.assertEqual(report["extra_paid_bytes"],0)
        self.assertTrue(report["native_ABI_unchanged"])
        self.assertTrue(np.array_equal(old.handles["b.weight"].packed,new.handles["b.weight"].packed))
        self.assertTrue(np.array_equal(old.handles["b.weight"].centroids,new.handles["b.weight"].centroids))
        self.assertTrue(np.array_equal(old.handles["b.weight"].row_scales,new.handles["b.weight"].row_scales))
        self.assertEqual(original[:5],rebuilt[:5])
        self.assertLessEqual(changed["rewrite_VALID_diagonal_weighted_MSE"],
                             changed["baseline_VALID_diagonal_weighted_MSE"]+1e-10)
        self.assertEqual(changed["physical_segment_bytes"],
                         sum(c[1] for c in new.meta["tensor_descriptors"]["a.weight"]["parts"].values()))

    def test_bad_activation_sizes_and_alias_rejected(self):
        source={"layer.weight":torch.randn(32,16)}
        original,_,_=encode(source)
        with self.assertRaises(AssertionError):
            reencode_8d(source["layer.weight"].numpy(),CB59(original).handles["layer.weight"],
                        np.ones(17),np.ones(16))
        with self.assertRaises(AssertionError):
            inplace_replace(original,"not_existing_block",{"indices":[]})

    def test_production_and_official_test_never_auto_authorized(self):
        for action in ("deploy","modify_jessica","merge","modify_main","use_frozen_TEST"):
            with self.assertRaises(ApprovalRequired):require_research_only(action)
        self.assertTrue(require_research_only("sandbox_rewrite"))
        d=decide_candidate(source_bytes=11229303,valid_teacher_KL_gain=1,
            valid_QA_regression=2,valid_greedy_ratio=.4,valid_speed_ratio=1.1)
        self.assertEqual(d["status"],"REJECT")
        self.assertFalse(d["automatic_merge_or_production_write"])
        self.assertTrue(physical_budget(11229303)["PASS"])

    def test_rank_truly_positive_causal_evidence(self):
        records={"single_module_FP32_oracle_rescue_not_paid_source":
           {"ranked_single_module_rescues":[
             {"module":"model.layers.13.self_attn.v_proj",
              "mean_KL_gain_from_exact_FP32_one_module":1.52},
             {"module":"model.layers.12.self_attn.v_proj",
              "mean_KL_gain_from_exact_FP32_one_module":.49}]}}
        self.assertEqual(priority_target(records),"model.layers.13.self_attn.v_proj")
        records["single_module_FP32_oracle_rescue_not_paid_source"]["ranked_single_module_rescues"]=[]
        with self.assertRaises(AssertionError):priority_target(records)

    def test_ingestion_fails_closed_for_official_TEST(self):
        payload={"experiment":"CALIBRI_v91_frozen_original_sources_root_cause_decomposition",
          "official_independent_TEST_never_loaded_or_tuned":False}
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"bad.json";p.write_text(json.dumps(payload))
            with self.assertRaises(AssertionError):ingest(p)

if __name__=="__main__":unittest.main()
