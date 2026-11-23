"""Z20 synthetic contract tests. Real metrics require already completed Z19 artifact.

No numerical tuning, no original43M forward, no DEV gold to inference.
"""
from __future__ import annotations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from research.calibri_nra3.s119q2i8z18_fresh28train14dev_multilayer_source import (
    BASELINE, ALTERNATE, rows, verify,
)
from research.calibri_nra3.s119q2i8z20_paired_consensus_readonly_forensics import (
    PHYSICAL_STAGE, PHYSICAL_REPORT, REPORT, STAGE, ARMS, MODES,
    paired_exact_discordance_p, wilson95, metrics_for_arm, run,
)


def synthetic_physical_records():
    """14 deterministic mock choices to verify pairing, not actual results."""
    examples=rows("DEV")
    out=[]
    for i,e in enumerate(examples):
        present=i<10
        f=i in (0,1,2)
        m=i in (0,1,3,4,5)
        final_id=100 if f else 210
        multi_id=100 if m else (210 if i>=6 else 220)
        choices={
            BASELINE:{
                "second_native_id":final_id,"correct":f,
                "changed_original_greedy":final_id!=400,
                "premature_EOS":False,
            },
            ALTERNATE:{
                "second_native_id":multi_id,"correct":m,
                "changed_original_greedy":multi_id!=400,
                "premature_EOS":False,
            },
        }
        out.append({
            "source_id":e["id"],"kind":e["kind"],"prompt_SHA256":f"mock{i:02d}",
            "own_FIRST_correct":True,
            "gold_SECOND_reachable_in_original_top64":present,
            "original_greedy_SECOND_correct":False,
            "original_greedy_SECOND_id":400,
            "choices":choices,
            "real_original43M_full_native_forwards":2,
            "new_Z18_DEV_now_SPENT_not_independent":True,
            "gold_only_posthoc_never_in_candidate_or_ranker_selection":True,
        })
    return out


def synthetic_physical_report():
    source=verify()
    originals=synthetic_physical_records()
    a={}
    for arm in ARMS:
        a[arm]={
            "new_DEV14_per_source":originals,
            "new_DEV14_after_training_once":{
                "own_FIRST_correct":14,
                "gold_SECOND_in_original_top64":10,
                "original_greedy_SECOND_correct":0,
                "same_source_matched_modes":{
                    BASELINE:{"correct_SECOND":3},
                    ALTERNATE:{"correct_SECOND":5},
                },
            }
        }
    return {
        "stage":PHYSICAL_STAGE,
        "precommitted_Z18_new_TRAIN28_SHA256":source["TRAIN_SHA256"],
        "precommitted_Z18_new_DEV14_SHA256":source["DEV_SHA256"],
        "DEV14_NOW_SPENT_not_independently_blind":True,
        "original43M_A_B_real_full_native_forwards":168,
        "same_native49152_topK":64,
        "same_per_shadow_trainable_params":1280,
        "no_DEV_gradient_or_gold_in_candidate_selection":True,
        "candidate_accepted":False,"output_gate":0,
        "TRUE_matched_A_B_SHADOW_FINAL_VS_MULTILAYER":a,
    }

class TestReadOnlyZ20(unittest.TestCase):
    def test_locked_old_spent_sha_and_stages(self):
        src=verify()
        self.assertEqual(src["TRAIN_SHA256"],
                         "7fe5c8111c10108a2332b1e0c82a08c23797c66620e64e3eff530bb365f88cb4")
        self.assertEqual(src["DEV_SHA256"],
                         "089e55c5dbd3b3a759c17f52e37217c25bb2e39e10a5d9e75c38c4fbd606fd86")
        self.assertIn("PAIRED_CONSENSUS_FORENSICS",STAGE)
        self.assertEqual(set(MODES),{BASELINE,ALTERNATE})
    def test_synthetic_paired_complete_and_true_oracle_limit(self):
        s=synthetic_physical_report()
        kinds={x["id"]:x["kind"] for x in rows("DEV")}
        result=metrics_for_arm(s["TRUE_matched_A_B_SHADOW_FINAL_VS_MULTILAYER"][ARMS[0]],kinds)
        counts=result["counts"]
        self.assertEqual(result["final_only_second_correct"],3)
        self.assertEqual(result["multilayer_second_correct"],5)
        self.assertEqual(result["paired_delta_multilayer_minus_final_correct"],2)
        self.assertEqual(counts["both_correct"],2)
        self.assertEqual(counts["final_only_correct"],1)
        self.assertEqual(counts["multilayer_only_correct"],3)
        self.assertEqual(counts["neither_correct"],8)
        self.assertEqual(result["posthoc_ORACLE_any_shadow_correct_upper_bound_NOT_a_rule"],6)
        self.assertEqual(result["candidate_coverage_upper_bound_original_top64"],10)
        self.assertEqual(result["paired_exact_two_sided_discordance_p_NOT_validation"],.625)
        self.assertEqual(result["source_n"],14)
        self.assertEqual(result["agreement_precision_POSTHOC_if_agree"],round(2/10,6))
        self.assertEqual(len(result["by_domain_2_sources_each"]),7)
        self.assertEqual(sum(r["source_n"] for r in result["by_domain_2_sources_each"].values()),14)
    def test_fail_if_same_selected_native_id_conflicts_gold_correctness(self):
        report=synthetic_physical_report()
        rows_source=report["TRUE_matched_A_B_SHADOW_FINAL_VS_MULTILAYER"][ARMS[0]]
        rows_source["new_DEV14_per_source"][2]["choices"][ALTERNATE]["second_native_id"]=100
        with self.assertRaises(ValueError):
            metrics_for_arm(rows_source,{x["id"]:x["kind"] for x in rows("DEV")})
    def test_source_only_run_preserves_original_and_refuses_overwrite(self):
        source=synthetic_physical_report()
        with TemporaryDirectory() as d:
            origin=Path(d)/PHYSICAL_REPORT
            origin.write_text(json.dumps(source),encoding="utf8")
            output=Path(d)/"z20_read_only"
            result=run(origin,output)
            self.assertTrue((output/REPORT).is_file())
            self.assertEqual(result["new_original43M_forwards_in_Z20"],0)
            self.assertEqual(result["new_training_or_fit_in_Z20"],0)
            self.assertEqual(result["candidate_accepted"],False)
            self.assertEqual(result["cross_A_B_source_overlap"]["FINAL_correct"]["A_and_B_correct"],3)
            self.assertEqual(json.loads(origin.read_text()),source)
            with self.assertRaises(ValueError):run(origin,output)
    def test_statistics_and_invalid_sources_fail_closed(self):
        self.assertEqual(wilson95(0,14)[0],0.)
        self.assertEqual(paired_exact_discordance_p(0,0),1.)
        self.assertEqual(paired_exact_discordance_p(1,3),.625)
        with self.assertRaises(ValueError):wilson95(17,14)
        source=synthetic_physical_report()
        source["original43M_A_B_real_full_native_forwards"]=12
        with TemporaryDirectory() as d:
            original=Path(d)/PHYSICAL_REPORT
            original.write_text(json.dumps(source),encoding="utf8")
            with self.assertRaises(ValueError):run(original,Path(d)/"out")
if __name__=="__main__":
    unittest.main()
