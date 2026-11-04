"""Small source-only tests; never open official v89 TEST."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from direct_codebook_v59 import QuantHandle,compress_matrix
from v117.protocol import ORIGINAL_CB59_SHA,MAX_SOURCE_BYTES
from v153.protocol import (check,select_layer,EXPECTED_BLOCK_MATRICES,
    NEXT_VALID,WIKI_NEXT_START,NEWS_NEXT_START)
from v153.fit import fit_one
from v153.codec import unpack


class PaidOriginalOperatorChecks(unittest.TestCase):
    def test_layer_selection_requires_all_original_30_FP32_oracle_controls(self):
        self.assertTrue(check())
        self.assertEqual(EXPECTED_BLOCK_MATRICES,7)
        self.assertEqual(NEXT_VALID,("wiki_paid_block_valid","news_paid_block_valid"))
        self.assertGreater(WIKI_NEXT_START,35840)
        self.assertGreater(NEWS_NEXT_START,11500)
        ranking=[
            {"layer":j,
             "mean_KL_delta_vs_original_CB59":-0.5 if j==13 else 1.}
            for j in range(30)]
        baseline={
          "stage":"v152_NONDEPLOYABLE_FP32_30BLOCK_CAUSAL_LOCALIZATION",
          "physically_packed_CB59_sha256":ORIGINAL_CB59_SHA,
          "summary":{"all_30_single_layer_FP32_oracle_ablation_ranking":
                     sorted(ranking,key=lambda x:x["mean_KL_delta_vs_original_CB59"])}
        }
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"oracle.json"
            path.write_text(json.dumps(baseline))
            idx,why=select_layer(path)
            self.assertEqual(idx,13)
            self.assertTrue(why["v152_unpaid_FP32_oracle_was_helpful"])
            baseline["summary"]["all_30_single_layer_FP32_oracle_ablation_ranking"].pop()
            path.write_text(json.dumps(baseline))
            with self.assertRaises(ValueError):
                select_layer(path)

    def test_one_physical_original_2D_matrix_residual_fit_without_teacher_logits(self):
        rng=np.random.default_rng(153)
        weight=rng.normal(size=(64,64)).astype(np.float32)
        packed=compress_matrix(weight)
        handle=QuantHandle(
            "model.layers.5.self_attn.q_proj.weight",(64,64),
            np.frombuffer(packed["packed"],dtype=np.uint8).copy(),
            np.frombuffer(packed["row_scales"],dtype="<f2").astype(np.float32),
            np.frombuffer(packed["centroids"],dtype="<f2").astype(np.float32).reshape(32,8))
        patch,metrics=fit_one(weight,handle)
        self.assertEqual(patch.name,handle.name)
        self.assertEqual(patch.shape,(64,64))
        self.assertEqual(patch.centers.shape,(32,2,8))
        self.assertEqual(len(patch.bits),(64*64//8+7)//8)
        self.assertTrue(metrics["all_source_weights_counted"])
        self.assertTrue(metrics["no_validation_or_QA_in_codebook_FIT"])
        self.assertTrue(np.isfinite(patch.centers).all())

    def test_corrupt_packed_full_source_fails_closed(self):
        with self.assertRaises(ValueError):
            unpack(b"not_a_complete_packed_source")
        self.assertGreater(MAX_SOURCE_BYTES,11_229_303)


if __name__=="__main__":
    unittest.main()
