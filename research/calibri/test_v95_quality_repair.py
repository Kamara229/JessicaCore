"""Ground truth behavior, same bytes, base runtime lock, no official TEST."""
import unittest
import torch
import numpy as np
from direct_codebook_v59 import encode,CB59
from v95.spec import verify,TRAIN_BOOLQ,TRAIN_SCIQ,VALID_BOOLQ,VALID_SCIQ
from v95.scoring import fit_rank
from v95.gates import decisions
from v95.variants import variants

def score(n,correct,balanced=.5):
    return {"score":{"count":n,"correct":correct,"balanced_accuracy":balanced}}

def result(b,s,distinct,repeat):
    return {"QA":{"BoolQ":score(32,b),"SciQ":score(24,s)},
        "greedy":{"mean_distinct_ratio":distinct,"mean_most_frequent_share":repeat}}

def valid_sample(b,s,balanced=.5):
    return {"BoolQ":score(64,b,balanced),"SciQ":score(32,s,balanced)}

class TestV95(unittest.TestCase):
    def test_splits_are_disjoint_from_all_previous_selection(self):
        self.assertTrue(verify())
        self.assertLess(TRAIN_BOOLQ[1],VALID_BOOLQ[0])
        self.assertLess(TRAIN_SCIQ[1],VALID_SCIQ[0])

    def test_fit_ranking_prefers_correct_answers_not_random_novelty(self):
        records={"bad_random":result(11,8,.98,.1),
                 "true_quality":result(17,13,.15,.8),
                 "underperform":result(12,9,.05,.9)}
        order,method=fit_rank(records)
        self.assertEqual(order[0],"true_quality")
        self.assertNotIn("KL",str(method))
        self.assertNotIn("NLL",str(method))

    def test_size_and_speed_lock_and_actual_ground_truth(self):
        candidate=valid_sample(34,18,.55)
        baseline=valid_sample(33,17,.50)
        teacher=valid_sample(30,18,.50)
        baseline_gen={"mean_distinct_ratio":.11,"mean_most_frequent_share":.83}
        candidate_gen={"mean_distinct_ratio":.24,"mean_most_frequent_share":.7}
        policy=decisions(candidate,baseline,teacher,11229303,11229303,
                         1.0,baseline_gen,candidate_gen)
        self.assertEqual(policy["decision"],"NEEDS_MORE_QUALITY_WORK")
        self.assertTrue(policy["checks"]["ground_truth_original_FP32_quality_preserved"])
        self.assertTrue(policy["checks"]["same_complete_archive_bytes"])
        self.assertFalse(policy["checks"]["greedy_distinct_ratio_at_least_0p25"])
        with self.assertRaises(AssertionError):
            decisions(candidate,baseline,teacher,11229304,11229303,
                      1.0,baseline_gen,candidate_gen)
        slow=decisions(candidate,baseline,teacher,11229303,11229303,
                      .88,baseline_gen,candidate_gen)
        self.assertFalse(slow["checks"]["full_model_native_speed_preserved_at_98pct"])

    def test_new_native_5bit_same_byte_variants_and_all_other_blocks_untouched(self):
        weights={"model.layers.13.self_attn.v_proj.weight":
                torch.randn(32,16,generator=torch.Generator().manual_seed(95)),
                "model.layers.0.mlp.down_proj.weight":
                torch.randn(16,16,generator=torch.Generator().manual_seed(96))}
        raw,_,_=encode(weights)
        old=CB59(raw)
        key="model.layers.13.self_attn.v_proj.weight"
        values=weights[key].numpy()
        # Self-controlled toy locked candidate == old; no active TEST,
        # no artifact mutation. Real source identical length in production.
        base,details=variants(raw,raw,values,
                {"FIT":np.linspace(1,4,16),"FIT_INTERNAL_HOLDOUT":np.linspace(1.2,3.7,16)})
        for name,blob in base.items():
            self.assertEqual(len(blob),len(raw))
            new=CB59(blob)
            other="model.layers.0.mlp.down_proj.weight"
            self.assertTrue(np.array_equal(old.handles[other].packed,new.handles[other].packed))
            self.assertTrue(np.array_equal(old.handles[other].centroids,new.handles[other].centroids))
            self.assertTrue(np.array_equal(old.handles[other].row_scales,new.handles[other].row_scales))
        self.assertEqual(len(base),5)

if __name__=="__main__":unittest.main()
