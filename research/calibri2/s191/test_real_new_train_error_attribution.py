"""S191 synthetic fail-closed posthoc diagnostic contracts (no weights)."""
import copy
import unittest
from calibri2.s191.real_new_train_error_attribution import diagnostic, EXPECTED_FAMILIES

def sample():
    rows=[]
    agg={}
    n=0
    for group,(count,expected_len) in EXPECTED_FAMILIES.items():
        for i in range(count):
            n+=1
            first=i<4
            rows.append({"prompt_sha256":f"{n:064x}","group":group,
                "generated_tokens":(expected_len-1 if first else expected_len),
                "eos_present":True,"first_token_correct":first,"full_EOS_correct":False})
        agg[group]={"count":8,"first_correct":4,"full_EOS_correct":0,"eos_present":8}
    return {"event":"S190_REAL_CPU_32_UNSEEN_NEW_TRAIN_PARAPHRASES_STRICT_SOURCE_ABSTAIN",
            "real_CPU_full_J7_new_TRAIN_all32":True,"original_real_CPU_FULL_EOS_456":456,
            "new128_real_CPU_FULL_EOS":128,"original_first_full_forgetting":0,
            "no_gold_or_SHA_at_inference_or_routing":True,
            "all_in_physical_bpp_charging_S188_S189_S190_source":0.6871,
            "model_accepted":False,"NEW_TRAIN_group_metrics":agg,
            "NEW_TRAIN_first_correct":16,"NEW_TRAIN_full_EOS_correct":0,
            "NEW_TRAIN_eos_present":32,"hashed_per_prompt_audit":rows}


class S191PosthocContracts(unittest.TestCase):
    def test_32_posthoc_errors_first_and_length(self):
        report=diagnostic(sample())
        self.assertEqual(report["new_train_full_EOS_correct"],0)
        self.assertEqual(report["failure_modes"],
                         {"first_right_wrong_output_length":16,"wrong_first_token":16})
        self.assertTrue(report["no_raw_prompt_or_gold_or_target_ids_in_output"])
        self.assertFalse(report["model_accepted"])

    def test_reject_forgetting_or_bpp_or_missing(self):
        for name,value in (("original_first_full_forgetting",1),
                           ("new128_real_CPU_FULL_EOS",127),
                           ("all_in_physical_bpp_charging_S188_S189_S190_source",.701)):
            r=sample()
            r[name]=value
            with self.assertRaises(AssertionError):
                diagnostic(r)
        r=sample()
        r["hashed_per_prompt_audit"].pop()
        with self.assertRaises(AssertionError):
            diagnostic(r)

    def test_reject_gold_or_metadata_corruption(self):
        r=sample()
        r["hashed_per_prompt_audit"][0]["full_EOS_correct"]=True
        with self.assertRaises(AssertionError):
            diagnostic(r)
        r=sample()
        r["hashed_per_prompt_audit"][0]["prompt_sha256"]="broken"
        with self.assertRaises(AssertionError):
            diagnostic(r)

if __name__ == "__main__":
    unittest.main()
