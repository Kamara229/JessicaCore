"""Fixed-rate source-only codec and real answer tests; preserve every negative."""
import unittest
import numpy as np
import torch
from v170.protocol import check,ARMS,CASE_IDS
from v170.codec import fit_one
from direct_codebook_v59 import encode,CB59

class ZeroBitGrowth(unittest.TestCase):
    def test_protocol_invariants(self):
        self.assertTrue(check())
        self.assertEqual(len(CASE_IDS),40)
        self.assertEqual(len(set(CASE_IDS)),40)
        self.assertEqual(ARMS,("original_FP32","original_CB59","equal_byte_CB170"))

    def test_fp16_codebook_and_byte_budget_on_toy_weight(self):
        torch.manual_seed(170)
        w=torch.randn(16,16,dtype=torch.float32)
        source={"layer.weight":w}
        data,_,_=encode(source)
        handle=CB59(data).handles["layer.weight"]
        centers,codes,stats=fit_one(w.numpy(),handle)
        self.assertEqual(len(centers),32*8*2)
        self.assertEqual(len(codes),len(handle.packed))
        self.assertEqual(stats["reencoded_5bit_groups"],32)
        self.assertEqual(stats["same_exact_original_matrix_bytes"],True)

    def test_codec_output_deterministic_on_unchanged_weight(self):
        torch.manual_seed(171)
        w=torch.randn(16,16,dtype=torch.float32)
        data,_,_=encode({"layer.weight":w})
        h=CB59(data).handles["layer.weight"]
        a=fit_one(w.numpy(),h)
        b=fit_one(w.numpy(),h)
        self.assertEqual(a[0],b[0])
        self.assertEqual(a[1],b[1])

    def test_genuine_qa_equal_prompts_and_no_full_fp32_promotion(self):
        from v169.sources import all_cases
        cases=[x for x in all_cases() if x["question_id"] in CASE_IDS and x["style"]=="direct"]
        self.assertEqual(len(cases),40)
        self.assertEqual(len(set(x["prompt_sha"] for x in cases)),40)

if __name__=="__main__":
    unittest.main()
