"""S151 orchestration guard must fail closed on missing joint/individual proof."""
import unittest
from calibri2.s151.real_three_on_one_host import candidate_is_safe,WORKERS

class S151ParallelIsolationTests(unittest.TestCase):
    def test_three_independent_non_duplicate_skills(self):
        self.assertEqual(len(WORKERS),3)
        self.assertEqual([r[1] for r in WORKERS].count("mps"),2)
        self.assertEqual([r[1] for r in WORKERS].count("cpu"),1)
        self.assertEqual(len({r[0] for r in WORKERS}),3)
    def test_quality_guard_fail_closed(self):
        self.assertFalse(candidate_is_safe(None))
        self.assertFalse(candidate_is_safe({}))
        candidate={
            "strict_no_forgetting":True,
            "TRAIN_only_skill_gain":True,
            "research_safe_development_candidate":True,
            "physical_safetensors_reloaded":True,
            "full456_and_new128":{
                "lost_old_full_ids":[],"lost_old_first_ids":[],
                "lost_new_full_ids":[],"lost_new_first_ids":[],
            },
            "physical":{"all_in_physical_bpp":0.68},
            "paired_CPU":{"ratio":0.95},
            "model_accepted":False
        }
        self.assertTrue(candidate_is_safe(candidate))
        candidate["full456_and_new128"]["lost_new_first_ids"]=["protected"]
        self.assertFalse(candidate_is_safe(candidate))
        candidate["full456_and_new128"]["lost_new_first_ids"]=[]
        candidate["TRAIN_only_skill_gain"]=False
        self.assertFalse(candidate_is_safe(candidate))
        candidate["TRAIN_only_skill_gain"]=True
        candidate["physical"]["all_in_physical_bpp"]=0.701
        self.assertFalse(candidate_is_safe(candidate))

if __name__=="__main__":
    unittest.main()
