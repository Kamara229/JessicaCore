"""Synthetic unit checks only. No model inference / QA reruns."""
from copy import deepcopy
import unittest
from v203.forensics import diagnose, EXPECTED_SOURCE_SHA


def accepted_row(index, keep=True):
    return {
        "step":index+1,"operator":"operator_"+str(index),
        "source_step_accepted":keep,
        "never_consulted_VALID_or_DEV_to_accept":True,
        "TRAIN_guard":{"ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE":keep},
        "per_step_TRAIN_sequence_guard":{
            "ACCEPT_FULL_SEQUENCE_TRAIN_ONLY":True,
            "adjacent_repeats_before":3,
            "adjacent_repeats_after":3,
            "correct_prefix_before":4,
            "correct_prefix_after":4,
        },
        "TRAIN_no_more_than_1percent_loss_increase":True,
    }


def _first_guard(fail=True):
    return {
        "ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE":not fail,
        "lost_previous_correct_TRAIN_first_tokens":[[0,"left"]] if fail else [],
        "lost_original_both_correct_opposite_TRAIN_pairs":[],
        "correct_by_domain_regressions":{},
        "diversity_regressed":False,
        "diversity_below_precommitted_floor":False,
    }


def _sequence_guard(fail=False):
    return {
        "ACCEPT_FULL_SEQUENCE_TRAIN_ONLY":not fail,
        "lost_original_complete_TRAIN_answers":[],
        "correct_prefix_before":5,"correct_prefix_after":4 if fail else 5,
        "prefix_regressed":fail,
        "adjacent_repeats_before":2,"adjacent_repeats_after":2,
        "repetition_regressed":False,"complete_exact_domain_regressions":{}
    }


def inputs():
    rows=[accepted_row(i, i<12) for i in range(16)]
    train={
        "actual_pilot_steps":16,"steps":rows,
        "trained_original_135M_unique_matrices":211,
        "QA_and_VALID_never_seen_for_TRAIN_acceptance":True,
        "all_pilot_mutations_rolled_back_on_full_TRAIN_fail":True,
        "accepted_steps_before_full_TRAIN_guard":12,
        "final_paid_5bit_net_changes":0,
        "final_FP16_centroid_net_cells_changed":0,
        "immutable_v186_full_TRAIN_guard":_first_guard(),
        "v202_immutable_greedy_full_sequence_TRAIN_guard":_sequence_guard()
    }
    seal={"v202_frozen_before_ANY_VALID_or_DEV":True,"sha256":EXPECTED_SOURCE_SHA}
    report={"source_SHA256":EXPECTED_SOURCE_SHA,"guard_accepted_step_count":12,
        "whole_original_source_bpp":0.6678394131307638,
        "final_guard_rolled_back":True}
    return train,seal,report


class TestGuardAudit(unittest.TestCase):
    def test_successfully_identifies_global_first_token_rollback(self):
        report=diagnose(*inputs())
        self.assertTrue(report["final_first_token_gate_failed"])
        self.assertFalse(report["final_sequence_gate_failed"])
        self.assertEqual(report["accepted_local_steps_but_all_reverted"],12)
        self.assertEqual(report["local_rejected_reasons"]["frozen_8pair_first_token_guard"],4)
        self.assertFalse(report["source_changed"])

    def test_sequence_only_global_failure_also_recorded(self):
        tr,seal,fin=inputs()
        tr["immutable_v186_full_TRAIN_guard"]=_first_guard(False)
        tr["v202_immutable_greedy_full_sequence_TRAIN_guard"]=_sequence_guard(True)
        result=diagnose(tr,seal,fin)
        self.assertTrue(result["final_sequence_gate_failed"])
        self.assertFalse(result["final_first_token_gate_failed"])
        self.assertTrue(result["full_frozen_8pair_sequence_failures"]["prefix_regressed"])

    def test_baseline_source_or_sealing_mismatch_rejected(self):
        a,b,c=inputs()
        b["sha256"]="invalid"
        with self.assertRaises(AssertionError):
            diagnose(a,b,c)
        a,b,c=inputs()
        b["v202_frozen_before_ANY_VALID_or_DEV"]=False
        with self.assertRaises(AssertionError):
            diagnose(a,b,c)

    def test_acceptance_and_step_drift_rejected(self):
        a,b,c=inputs()
        a["steps"][0]["source_step_accepted"]=False
        with self.assertRaises(AssertionError):
            diagnose(a,b,c)
        a,b,c=inputs()
        a["steps"][1]["operator"]=a["steps"][0]["operator"]
        with self.assertRaises(AssertionError):
            diagnose(a,b,c)

    def test_unexpected_all_global_pass_is_inconsistent(self):
        a,b,c=inputs()
        a["immutable_v186_full_TRAIN_guard"]=_first_guard(False)
        a["v202_immutable_greedy_full_sequence_TRAIN_guard"]=_sequence_guard(False)
        with self.assertRaises(AssertionError):
            diagnose(a,b,c)


if __name__=="__main__":
    unittest.main()
