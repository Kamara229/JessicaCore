"""Synthetic tests for v210 public QA artifact invariants. No model execution."""
import copy
import unittest
from v210.read_only import analyze

def qa(version):
    statuses={"FP32_existing_135M":["machine_correct"]*6+["machine_wrong"]*6+["ungraded"]*4,
              "CB206":["machine_correct"]*3+["machine_wrong"]*10+["ungraded"]*3,
              "GT206":["machine_correct"]*3+["machine_wrong"]*10+["ungraded"]*3,
              "GT208":["machine_correct"]*2+["machine_wrong"]*10+["ungraded"]*4}
    scores={}
    for key in (["FP32_existing_135M","GT206"] if version==206 else
                ["FP32_existing_135M","CB206","GT208"]):
        seq=statuses[key]
        rows=[{"case_id":i,"domain":"x","machine_status":s,
                "raw":("frozen-"+str(i) if key=="FP32_existing_135M" else
                "v206-"+str(i) if key in ("GT206","CB206") else
                "v208-"+str(i))} for i,s in enumerate(seq)]
        scores[key]={"n":16,"machine_confirmed":seq.count("machine_correct"),
          "machine_recognized_wrong":seq.count("machine_wrong"),
          "ungraded_not_known_wrong":seq.count("ungraded"),
          "verbatim_answers_and_grades":rows}
    return {"cases":[{"case_id":i,"gold":"x","question":str(i)} for i in range(16)],
            "preexisting_v192_frozen_grader_sha256":"locked","scores":scores}

def audit(offset):
    rows=[{"step":i+1,"operator":str(offset+i),
      "source_step_accepted":i%4!=0,
      "TRAIN_guard":{"ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE":True},
      "per_step_TRAIN_full_24_guard":{"ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE":True},
      "per_step_TRAIN_sequence_guard":{"ACCEPT_FULL_SEQUENCE_TRAIN_ONLY":True},
      "TRAIN_no_more_than_1percent_loss_increase":True} for i in range(48)]
    return {"steps":rows,"actual_pilot_steps":48,
            "accepted_steps_before_full_TRAIN_guard":36,
            "all_pilot_mutations_rolled_back_on_full_TRAIN_fail":False}

def summary(v):
    return {"completed":True,"whole_original_source_bpp":.66783941313,
            "native_CPU_speed_ratio_CB59":.99,
            "source_SHA256":("5a6d5f07f9e49fa6063dd772d7e2aee53beeb3527d10d5a429715593b68939de"
                             if v==206 else
                             "cd0947dcd34b1befee8aa2d58821b869a8476e8156a78a2a9747ea565f5b90ec")}

class TestV210(unittest.TestCase):
    def run_case(self,q0=None,q1=None,a0=None,a1=None):
        return analyze(q0 or qa(206),q1 or qa(208),a0 or audit(0),
                       a1 or audit(48),summary(206),summary(208))

    def test_matched_original_but_negative_compressed_qa(self):
        result=self.run_case()
        self.assertEqual((result["FP32_confirmed"],result["v206_confirmed"],
                          result["v208_confirmed"]),(6,3,2))
        self.assertEqual(result["statuses_including_ungraded"],{
            "machine_correct -> machine_correct":2,
            "machine_correct -> machine_wrong":1,
            "machine_wrong -> machine_wrong":9,
            "machine_wrong -> ungraded":1,
            "ungraded -> ungraded":3})
        self.assertTrue(result["PUBLIC_DEV_not_blind_and_NOT_checkpoint_training_parameter_selector"])

    def test_original_fp32_text_drift_fails(self):
        x=qa(208)
        x["scores"]["FP32_existing_135M"]["verbatim_answers_and_grades"][0]["raw"]="drift"
        with self.assertRaises(AssertionError):self.run_case(q1=x)

    def test_v206_baseline_text_drift_fails(self):
        x=qa(208)
        x["scores"]["CB206"]["verbatim_answers_and_grades"][0]["raw"]="drift"
        with self.assertRaises(AssertionError):self.run_case(q1=x)

    def test_repeated_operator_hard_fail(self):
        with self.assertRaises(AssertionError):self.run_case(a1=audit(24))

    def test_grader_drift_fails(self):
        x=qa(208);x["preexisting_v192_frozen_grader_sha256"]="drift"
        with self.assertRaises(AssertionError):self.run_case(q1=x)

if __name__=="__main__":unittest.main()
