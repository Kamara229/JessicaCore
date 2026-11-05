"""No pretrained model download: prove universal schedule and strict FP32 gate."""
import hashlib
import unittest

from green_tunnel import PhysicalSource,verify_physical_source,require_paid_update
from green_tunnel.schedule import Operator,plan_cycles
from green_tunnel.parity import compare_machine_parity


def toy_source(binary=b"abcdefg"):
    return PhysicalSource("TEST-only 5-bit codec",100,len(binary),
          hashlib.sha256(binary).hexdigest(),aliases=(("lm_head.weight","embeddings.weight"),))


def arithmetic_cases(n=96):
    cases=[]
    for i in range(n):
        a=300+i;b=51+i
        cases.append({"case_id":i,"domain":"arithmetic",
          "question":f"Compute {a} plus {b}. Reply with the result as one number.",
          "gold":str(a+b)})
    return cases


class PhysicalBudgetContractTest(unittest.TestCase):
    def test_actual_binary_signed_bpp_and_aliases(self):
        origin=toy_source()
        verified=verify_physical_source(
           origin,b"abcdefg",lambda x:{"ids":(1,2)},
           lambda obj,aliases: bool(obj["ids"] and aliases))
        self.assertAlmostEqual(verified["actual_complete_bpp"],0.56)
        self.assertTrue(verified["aliases_verified"])
        with self.assertRaises(ValueError):
            verify_physical_source(origin,b"abcdefh")
        with self.assertRaises(AssertionError):
            require_paid_update(origin,PhysicalSource(
                origin.codec,origin.original_unique_parameters,9,
                hashlib.sha256(b"123456789").hexdigest(),aliases=origin.aliases
            ),b"123456789")
        with self.assertRaises(AssertionError):
            require_paid_update(origin,PhysicalSource(
                origin.codec,origin.original_unique_parameters,7,
                hashlib.sha256(b"newbits").hexdigest(),aliases=()
            ),b"newbits")
        signed=require_paid_update(origin,toy_source(b"xxxxxxx"),b"xxxxxxx")
        self.assertEqual(signed["new_green_tunnel_inference_weights_outside_binary"],0)

    def test_never_allow_zero_size_fake_model_or_unpaid_bits(self):
        with self.assertRaises(ValueError):
            PhysicalSource("CB",0,1,"a"*64)
        with self.assertRaises(ValueError):
            PhysicalSource("CB",100,1,"xxxx")
        with self.assertRaises(ValueError):
            PhysicalSource("CB",100,1,"a"*64,maximum_bpp=.71)


class UniversalSourceScheduleTest(unittest.TestCase):
    def test_exact_all_unique_211_sources_every_cycle_no_hidden_special_names(self):
        fake=[
           Operator(f"decoder.stack_{i}.attention.weight","attention",i,1000)
           for i in range(105)]
        fake += [
           Operator(f"experts.stacks_{i}.ffn.expert{i}.weight","expert_mlp",i,1000)
           for i in range(106)]
        rounds=plan_cycles(fake,3)
        self.assertEqual(len(rounds),3)
        self.assertTrue(all(len(r)==211 and len(set(r))==211 for r in rounds))
        ids={x.canonical_weight_name for x in fake}
        self.assertTrue(all(set(r)==ids for r in rounds))
        self.assertNotEqual(rounds[0],rounds[1])
        self.assertNotEqual(rounds[1],rounds[2])
        with self.assertRaises(ValueError):
            plan_cycles(fake+[fake[0]],2)
        with self.assertRaises(ValueError):
            plan_cycles([],2)
        with self.assertRaises(ValueError):
            plan_cycles(fake,0)


class FullAnswerParityTest(unittest.TestCase):
    def setUp(self):
        self.examples=arithmetic_cases(96)
        self.correct={c["case_id"]:c["gold"] for c in self.examples}
        self.source=toy_source()
    def result(self,replies,**kw):
        args={"paired_native_cpu_ratio":.95,
              "qa_independent_of_training_and_model_selection":True,
              "both_checkpoints_frozen_before_QA":True,
              "native_only_paid_weights":True,
              "native_eos_count":96,
              "full_answer_distinct_fraction":.5}
        args.update(kw)
        return compare_machine_parity(self.examples,self.correct,replies,
                        self.source,**args)
    def test_retain_all_original_correct_AND_independent_96_CPU_rate(self):
        r=self.result(self.correct)
        self.assertTrue(r["CERTIFIED_PARITY_TO_ORIGINAL_FP32_FOR_SCALING"])
        self.assertEqual(r["FP32_correct_cases_LOST_in_CALIBRI"],[])
        self.assertEqual(r["original_machine_confirmed"],96)
        self.assertTrue(r["native_90pct_median_speed_pass"])
    def test_exposes_one_lost_correct_answer_even_if_aggregate_wrong(self):
        replies=dict(self.correct);replies[12]="not a number"
        r=self.result(replies)
        self.assertFalse(r["CERTIFIED_PARITY_TO_ORIGINAL_FP32_FOR_SCALING"])
        self.assertIn(12,r["FP32_correct_cases_LOST_in_CALIBRI"])
    def test_gate_requires_untouched_test_and_real_cpu_measurement(self):
        r=self.result(self.correct,
             qa_independent_of_training_and_model_selection=False)
        self.assertFalse(r["CERTIFIED_PARITY_TO_ORIGINAL_FP32_FOR_SCALING"])
        self.assertIn("not_presealed_sufficient_independent_QA",r["FAIL_OR_UNDETERMINED_REASONS"])
        r=self.result(self.correct,paired_native_cpu_ratio=None)
        self.assertFalse(r["CERTIFIED_PARITY_TO_ORIGINAL_FP32_FOR_SCALING"])
        r=self.result(self.correct,paired_native_cpu_ratio=.89)
        self.assertFalse(r["native_90pct_median_speed_pass"])
        r=self.result(self.correct, native_only_paid_weights=False)
        self.assertFalse(r["CERTIFIED_PARITY_TO_ORIGINAL_FP32_FOR_SCALING"])
        r=self.result(self.correct, native_eos_count=0)
        self.assertFalse(r["CERTIFIED_PARITY_TO_ORIGINAL_FP32_FOR_SCALING"])
    def test_all_yes_or_all_ungraded_cannot_masquerade_as_fp32_parity(self):
        r=self.result({c["case_id"]:"Yes" for c in self.examples})
        self.assertEqual(r["calibri_machine_confirmed"],0)
        self.assertEqual(len(r["FP32_correct_cases_LOST_in_CALIBRI"]),96)
        self.assertFalse(r["CERTIFIED_PARITY_TO_ORIGINAL_FP32_FOR_SCALING"])
    def test_small_dev_set_cannot_certify_parity_even_when_all_answer_right(self):
        subset=self.examples[:16]
        old={i:self.correct[i] for i in range(16)}
        r=compare_machine_parity(subset,old,old,self.source,
            paired_native_cpu_ratio=1.0,
            qa_independent_of_training_and_model_selection=True,
            both_checkpoints_frozen_before_QA=True,
            native_only_paid_weights=True)
        self.assertTrue(r["observed_development_MACHINE_parity_not_human"])
        self.assertFalse(r["CERTIFIED_PARITY_TO_ORIGINAL_FP32_FOR_SCALING"])


if __name__=="__main__":
    unittest.main()
