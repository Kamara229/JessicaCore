"""Pure CPU deterministic v91 unit tests; no model downloads."""
import unittest
import numpy as np
import torch
from v91.heads import HeadComponents,category_margin,summarize
from v91.layers import numerics
from v91.influence import compare_logits,candidates_by_local_error
from v88.tensor import delta,normalized,OUT1,OUT2,IN1,IN2,RANK,VOCAB

class TestV91(unittest.TestCase):
    def test_paid_tensor_selected_token_matches_real_full_operator(self):
        gen=np.random.default_rng(91)
        A=gen.normal(0,0.02,(OUT1,IN1,RANK)).astype("f2")
        B=gen.normal(0,0.02,(RANK,OUT2,IN2)).astype("f2")
        b=np.zeros(VOCAB,dtype="f2")
        x=torch.randn(IN1*IN2, generator=torch.Generator().manual_seed(91))
        obj=HeadComponents(A,B,b)
        ref=delta(normalized(x),obj.A,obj.B)
        for token in (0,1,345,10056,49151):
            self.assertAlmostEqual(obj.token_tensor(x,token),float(ref[token]),places=5)

    def test_bias_and_tensor_yes_no_margin_decomposition(self):
        gen=np.random.default_rng(92)
        A=gen.normal(0,0.03,(OUT1,IN1,RANK)).astype("f2")
        B=gen.normal(0,0.03,(RANK,OUT2,IN2)).astype("f2")
        bias=np.zeros(VOCAB,dtype="f2")
        bias[7]=1.5
        bias[9]=-0.5
        logits=torch.zeros(VOCAB)
        logits[7]=2
        logits[9]=1
        c=HeadComponents(A,B,bias).margins(torch.ones(IN1*IN2),logits,(7,9))
        self.assertAlmostEqual(c["CB59"],1)
        self.assertAlmostEqual(c["bias_yes_minus_no"],2)
        self.assertAlmostEqual(c["bias_only"],3)
        self.assertAlmostEqual(c["CB88"]-c["tensor_only"],2,places=5)

    def test_binary_confusion_handles_class_imbalance(self):
        row={"gold":0,"modes":{},"bias_yes_minus_no":-2.0,"tensor_yes_minus_no":-1.0}
        row["modes"]={key:category_margin(margin,0) for key,margin
                      in zip(("CB59","bias_only","tensor_only","CB88"),(1,-1,0.5,-2))}
        s=summarize([row])
        self.assertEqual(s["CB59"]["correct_yes"],1)
        self.assertEqual(s["CB88"]["correct_yes"],0)
        self.assertEqual(s["CB88"]["predicted_yes"],0)

    def test_rms_distinguishes_teacher_local_deviation(self):
        a=torch.ones(8)
        self.assertAlmostEqual(numerics(a,a)["relative_RMS_error"],0)
        self.assertAlmostEqual(numerics(a,a*2)["relative_RMS_error"],1)

    def test_oracle_kl_identical_logits_zero(self):
        t=torch.tensor([1.,2.,4.])
        self.assertAlmostEqual(compare_logits(t,t,(0,1))["teacher_student_KL"],0,places=6)

    def test_ranked_causal_candidates_include_head_and_early_late(self):
        local={f"model.layers.{j}.mlp.down_proj":{"relative_RMS_error":float(j)/30}
               for j in range(30)}
        picks=candidates_by_local_error(local)
        self.assertIn("lm_head",picks)
        self.assertIn("model.embed_tokens",picks)
        self.assertIn("model.layers.0.mlp.down_proj",picks)
        self.assertIn("model.layers.29.mlp.down_proj",picks)
        self.assertLessEqual(len(picks),10)

if __name__=="__main__":unittest.main()
