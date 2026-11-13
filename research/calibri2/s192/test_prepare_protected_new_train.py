"""S192 source-only deterministic readiness contract, no training or model calls."""
import copy
import unittest

from calibri2.s192.prepare_protected_new_train import (
    verify_source_readiness,digest,SUPPORTED_GROUPS,
)
from calibri2.s190.real_cpu_new_train_paraphrase_probe import new_train_paraphrase_probes

class FakeTok:
    eos_token_id=0
    def encode(self,answer,add_special_tokens=False):
        if answer in ("Yes","No"):
            return [1]
        return [2,3] if len(answer.split())==2 else []

def fixture():
    orig={digest(f"original/{i}") for i in range(456)}
    new={digest(f"S122/{i}") for i in range(128)}
    probes=new_train_paraphrase_probes()
    audit=[{"prompt_sha256":digest(q),"group":family,
            "full_EOS_correct":False,"first_token_correct":i%2==0}
           for i,(q,_,family) in enumerate(probes)]
    m={
        "event":"S190_REAL_CPU_32_UNSEEN_NEW_TRAIN_PARAPHRASES_STRICT_SOURCE_ABSTAIN",
        "original_real_CPU_FULL_EOS_456":456,
        "new128_real_CPU_FULL_EOS":128,
        "original_first_full_forgetting":0,
        "real_CPU_full_J7_new_TRAIN_all32":True,
        "no_gold_or_SHA_at_inference_or_routing":True,
        "no_neural_weights_updated":True,
        "model_accepted":False,
        "all_in_physical_bpp_charging_S188_S189_S190_source":0.687094951021114,
        "original_CPU_S101_ratio":2.46694,
        "NEW_TRAIN_full_EOS_correct":0,
        "NEW_TRAIN_first_correct":16,
        "hashed_per_prompt_audit":audit,
    }
    n={
        "stage":"S191_POSTHOC_REAL_S190_NEW_TRAIN_FAILURE_MODES_NO_NEURAL",
        "S190_immutable_run":43340296813,
        "new_train_probe_count":32,
        "new_train_full_EOS_correct":0,
        "new_train_first_correct":16,
        "not_a_new_neural_generalization_result":True,
        "model_accepted":False,
        "failure_modes":{"wrong_first_token":16,"first_right_wrong_continuation_token":16},
    }
    s122digest=digest("\n".join(sorted(new)))
    return orig,new,m,n,s122digest


class S192ReadinessContracts(unittest.TestCase):
    def test_authorized_456_plus_128_plus32_all_unique(self):
        orig,new,m,n,d=fixture()
        result=verify_source_readiness(FakeTok(),orig,new,m,n,d,charge_bytes=10000)
        self.assertEqual(result["new_TRAIN_paraphrase_32_count"],32)
        self.assertEqual(result["full_EOS_and_first_token_protected_584"],584)
        self.assertEqual(result["NEW_TRAIN_per_family"],SUPPORTED_GROUPS)
        self.assertEqual(result["NEW_TRAIN_target_lengths_including_EOS"],
                         {2:8,3:24})
        self.assertFalse(result["training_executed"])
        self.assertFalse(result["new_weights_emitted"])
        self.assertFalse(result["model_accepted"])
        self.assertTrue(result["training_contract"]["atomic_weights_and_Adam_rollback_on_any_failed_gate"])
        self.assertLessEqual(result["pretraining_max_bpp_including_research_code"],.7)

    def test_reject_collision_with_old_or_protected_new(self):
        orig,new,m,n,d=fixture()
        victim=digest(new_train_paraphrase_probes()[0][0])
        orig.discard(next(iter(orig)))
        orig.add(victim)
        with self.assertRaisesRegex(AssertionError,"collision"):
            verify_source_readiness(FakeTok(),orig,new,m,n,d,charge_bytes=0)
        orig,new,m,n,d=fixture()
        new.discard(next(iter(new)))
        new.add(victim)
        d=digest("\n".join(sorted(new)))
        with self.assertRaisesRegex(AssertionError,"collision"):
            verify_source_readiness(FakeTok(),orig,new,m,n,d,charge_bytes=0)

    def test_reject_changed_sha_to_family_and_incomplete_new32(self):
        for action in ("wrong_family","missing"):
            orig,new,m,n,d=fixture()
            if action=="wrong_family":
                m["hashed_per_prompt_audit"][0]["group"]="numeric_new_wording"
            else:
                m["hashed_per_prompt_audit"].pop()
            with self.assertRaises(AssertionError):
                verify_source_readiness(FakeTok(),orig,new,m,n,d,charge_bytes=0)

    def test_reject_wrong_tokenizer_provenance_and_bpp(self):
        orig,new,m,n,d=fixture()
        with self.assertRaisesRegex(AssertionError,"digest"):
            verify_source_readiness(FakeTok(),orig,new,m,n,"f"*64,charge_bytes=0)
        with self.assertRaisesRegex(AssertionError,"bpp"):
            verify_source_readiness(FakeTok(),orig,new,m,n,d,charge_bytes=100000)

    def test_reject_missing_old_or_new_guards(self):
        for key,value in (("original_real_CPU_FULL_EOS_456",455),
                          ("new128_real_CPU_FULL_EOS",127),
                          ("original_first_full_forgetting",1),
                          ("no_gold_or_SHA_at_inference_or_routing",False),
                          ("model_accepted",True)):
            orig,new,m,n,d=fixture()
            m[key]=value
            with self.subTest(field=key),self.assertRaises(AssertionError):
                verify_source_readiness(FakeTok(),orig,new,m,n,d,charge_bytes=0)


if __name__=="__main__":
    unittest.main()
