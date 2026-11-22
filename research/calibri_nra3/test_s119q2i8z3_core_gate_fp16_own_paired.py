"""S119q2i8z3 actual Z2 native49k pretrained 43M A/B paired read-only audit."""
import unittest
from research.calibri_nra3.s119q2i8z3_core_gate_fp16_own_paired_audit import (
    STAGE,paired_own,paired_gold_second,run,VOCAB,
)
def sample(id,ids):
    return {"source_id":id,"generated_own_native_BPE_ids":ids,
            "source_internal_DEV_NOT_independent_blind":True,
            "true_native_full_EOS_correct":False,
            "first_two_gold_BPE_correct":False,
            "first_three_gold_BPE_correct":False,
            "native_EOS":bool(ids and ids[-1]==2)}
class TestZ2CompletedPair(unittest.TestCase):
    def test_01_paired_true49k_native_BPE(self):
        a={"NEW_DEV":{"OWN_samples":[sample("a",[17,18]),sample("b",[11,2])]}}
        b={"NEW_DEV":{"OWN_samples":[sample("a",[17,19]),sample("b",[11,2])]}}
        p=paired_own(a,b,"NEW_DEV",2)
        self.assertEqual(p["source_count"],2)
        self.assertEqual(p["different_A_B_real_OWN_BPE_sequences"],1)
        self.assertEqual(p["paired_source_rows"][0][
            "first_different_generated_native_BPE_position_1based"],2)
    def test_02_no_false_paired_source_identity(self):
        a={"NEW_DEV":{"OWN_samples":[sample("a",[13])]}}
        b={"NEW_DEV":{"OWN_samples":[sample("b",[13])]}}
        with self.assertRaises(ValueError):paired_own(a,b,"NEW_DEV",1)
    def test_03_full49k_paired_rank_reject_bad(self):
        def rows(rank):
            return {"per_source_read_only_teacherforced_previous_gold_BPE":[{"source_id":str(i),
                "correct_gold_second_native_rank_full49152":rank,
                "given_gold_first_BPE_ONLY_for_readonly_diagnostic":True}
                for i in range(14)]}
        r=paired_gold_second(rows(20),rows(15))
        self.assertEqual(r["margin_B_better_lower_gold_rank_than_A"],14)
        self.assertEqual(r["margin_B_worse_higher_gold_rank_than_A"],0)
    def test_04_strict_source_provenance(self):
        self.assertIn("FP16_CORE_GATE_L2",STAGE)
        self.assertEqual(VOCAB,49152)
        with self.assertRaises(Exception):run({},None)
if __name__=="__main__":unittest.main()
