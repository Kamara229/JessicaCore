"""Synthetic no-model regression tests for v207 retrospective immutable artifact study."""
import copy
import unittest
from v207.read_only import analyze


def qa(revision):
    cases=[{"case_id":i,"question":str(i),"gold":"x","domain":"arithmetic"} for i in range(16)]
    if revision == 204:
        statuses=["machine_correct"]*2+["machine_wrong"]*6+["ungraded"]*8
    else:
        statuses=["machine_correct"]*3+["machine_wrong"]*10+["ungraded"]*3
    src={"FP32_existing_135M":["machine_correct"]*6+["machine_wrong"]*6+["ungraded"]*4,
         "CB186":["machine_correct"]*2+["ungraded"]*14,
         "GT"+str(revision):statuses}
    scores={}
    for label,seq in src.items():
        rows=[{"case_id":i,"domain":"arithmetic","machine_status":s,
               "raw":("frozen-"+label+"-"+str(i)) if label!="GT"+str(revision)
                     else (f"source{revision}-{i}")} for i,s in enumerate(seq)]
        scores[label]={"n":16,"machine_confirmed":seq.count("machine_correct"),
          "machine_recognized_wrong":seq.count("machine_wrong"),
          "ungraded_not_known_wrong":seq.count("ungraded"),
          "verbatim_answers_and_grades":rows}
    return {"cases":cases,"preexisting_v192_frozen_grader_sha256":"frozen-v192",
            "scores":scores}


def audit(n):
    guard={"ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE":True}
    seq={"ACCEPT_FULL_SEQUENCE_TRAIN_ONLY":True}
    rows=[{"step":i,"operator":f"layer.{i}",
           "source_step_accepted":i%4!=0,
           "TRAIN_guard":guard,"per_step_TRAIN_full_24_guard":guard,
           "per_step_TRAIN_sequence_guard":seq,
           "TRAIN_no_more_than_1percent_loss_increase":True}
          for i in range(1,n+1)]
    return {"steps":rows,"actual_pilot_steps":n,
            "accepted_steps_before_full_TRAIN_guard":sum(r["source_step_accepted"] for r in rows),
            "all_pilot_mutations_rolled_back_on_full_TRAIN_fail":False}


def summary(version):
    return {"completed":True,"source_SHA256":str(version)*32,
            "whole_original_source_bpp":.6678394131307638,
            "native_CPU_speed_ratio_CB59":1.,
            "original_FP32_16_machine_correct":6}


class ReadOnlyAnalysisTests(unittest.TestCase):
    def evaluate(self, q204=None, q206=None, a204=None, a206=None):
        return analyze(q204 or qa(204),q206 or qa(206),
                       a204 or audit(16),a206 or audit(48),summary(204),summary(206))

    def test_immutable_three_way_full_answers_and_ungraded(self):
        result=self.evaluate()
        self.assertEqual([result[k] for k in ("FP32_confirmed_correct",
            "v186_confirmed_correct","v204_confirmed_correct",
            "v206_confirmed_correct")],[6,2,2,3])
        self.assertEqual(result["status_transitions_ungraded_not_equal_wrong"],
            {"machine_correct -> machine_correct":2,
             "machine_wrong -> machine_correct":1,
             "machine_wrong -> machine_wrong":5,
             "ungraded -> machine_wrong":5,
             "ungraded -> ungraded":3})
        self.assertEqual(len(result["v206_TRAIN"]["accepted"]),36)
        self.assertTrue(result["PUBLIC_DEV_NOT_used_to_accept_changes_choose_checkpoints_or_hyperparameters"])

    def test_gold_case_drift_hard_fail(self):
        b=qa(206)
        b["cases"][0]["gold"]="changed"
        with self.assertRaises(AssertionError): self.evaluate(q206=b)

    def test_original_FP32_drift_hard_fail(self):
        b=qa(206)
        b["scores"]["FP32_existing_135M"]["verbatim_answers_and_grades"][1]["raw"]="changed"
        with self.assertRaises(AssertionError): self.evaluate(q206=b)

    def test_grader_drift_hard_fail(self):
        b=qa(206)
        b["preexisting_v192_frozen_grader_sha256"]="changed"
        with self.assertRaises(AssertionError): self.evaluate(q206=b)

    def test_early_operator_schedule_drift_hard_fail(self):
        b=audit(48)
        b["steps"][0]["operator"]="different"
        with self.assertRaises(AssertionError):self.evaluate(a206=b)

    def test_rejected_guard_reasons_are_not_exclusive(self):
        b=audit(48)
        b["steps"][3]["TRAIN_guard"]={"ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE":False}
        b["steps"][3]["per_step_TRAIN_full_24_guard"]={"ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE":False}
        b["steps"][3]["per_step_TRAIN_sequence_guard"]={"ACCEPT_FULL_SEQUENCE_TRAIN_ONLY":False}
        result=self.evaluate(a206=b)
        self.assertEqual(result["v206_TRAIN"]["rejected_guard_reasons_overlapping"],
                         {"global_24_first":1,"global_8_full_sequences":1,"sampled_8_first":1})

if __name__=="__main__":
    unittest.main()
