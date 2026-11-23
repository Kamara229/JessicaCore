"""Z11 physical source-only strict native self-BPE and rank forensic unit tests."""
import unittest
from research.calibri_nra3.s119q2i8z11_real_fp16_own_paired_audit import (
    STAGE, A, B, VOCAB, paired_own, paired_gold_second, run,
)

def sample(source, ids, arm):
    return {
        "source_id":source, "original_prompt_SHA256":"sha_"+source,
        "generated_own_native_BPE_ids":ids, "eval_arm":arm,
        "source_internal_DEV_NOT_independent_blind":True,
        "true_native_full_EOS_correct":False,
        "first_gold_BPE_correct":True,
        "first_two_gold_BPE_correct":False,
        "first_three_gold_BPE_correct":False,
        "native_EOS":bool(ids[-1]==2),
        "valid_full_UTF8_bytes":False,
    }

def gold(rank, nll, source):
    return {
        "source_id":source,"source_prompt_sha256":"sha_"+source,
        "correct_gold_second_native_rank_full49152":rank,
        "gold_second_NLL_nats":nll,
        "given_gold_first_BPE_ONLY_for_readonly_diagnostic":True,
    }

class TestZ11RealSourceOnly(unittest.TestCase):
    def test_stage_native_49152(self):
        self.assertIn("FP16_OWN_PAIRED",STAGE)
        self.assertEqual(VOCAB,49152)
        with self.assertRaises(Exception):run({},None)

    def test_strict_same_source_OWN_and_divergence(self):
        x={"NEW_DEV":{"OWN_samples":[sample("a",[17,27,2],A),sample("b",[15,32],A)]}}
        y={"NEW_DEV":{"OWN_samples":[sample("a",[17,28,2],B),sample("b",[15,32],B)]}}
        result=paired_own(x,y,"NEW_DEV",2)
        self.assertEqual(result["same_A_B_full_OWN_BPE_sequences"],1)
        self.assertEqual(result["different_A_B_full_OWN_BPE_sequences"],1)
        self.assertEqual(result["paired_sample_details"][0]["first_A_vs_B_generated_BPE_divergence_1based"],2)

    def test_own_source_MUST_match_and_be_spent(self):
        x={"NEW_DEV":{"OWN_samples":[sample("a",[17,27,2],A)]}}
        y={"NEW_DEV":{"OWN_samples":[sample("b",[17,28,2],B)]}}
        with self.assertRaises(ValueError):paired_own(x,y,"NEW_DEV",1)
        z=sample("a",[17,28,2],B)
        z["source_internal_DEV_NOT_independent_blind"]=False
        with self.assertRaises(ValueError):
            paired_own(x,{"NEW_DEV":{"OWN_samples":[z]}},"NEW_DEV",1)

    def test_gold_previous_rank_and_NLL_paired_independent_of_OWN(self):
        a={"per_source_read_only_teacherforced_previous_gold_BPE":[
            gold(100,5.2,str(i)) for i in range(14)]}
        b={"per_source_read_only_teacherforced_previous_gold_BPE":[
            gold(98,5.1,str(i)) for i in range(14)]}
        r=paired_gold_second(a,b)
        self.assertEqual(r["B_gold_rank_better"],14)
        self.assertEqual(r["B_gold_NLL_better"],14)
        self.assertAlmostEqual(r["mean_B_minus_A_gold_NLL_nats"],-.1)

    def test_reject_wrong_previous_gold_identity(self):
        a={"per_source_read_only_teacherforced_previous_gold_BPE":[
            gold(100,5.2,str(i)) for i in range(14)]}
        b={"per_source_read_only_teacherforced_previous_gold_BPE":[
            gold(98,5.1,str(i)) for i in range(14)]}
        b["per_source_read_only_teacherforced_previous_gold_BPE"][0][
            "source_prompt_sha256"]="other"
        with self.assertRaises(ValueError):paired_gold_second(a,b)

if __name__=="__main__":
    unittest.main()
