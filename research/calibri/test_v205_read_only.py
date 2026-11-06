"""No model runs: synthetic strict immutable v200/v204 comparison checks."""
import unittest
from v205.read_only import analyze


def qa(new):
    statuses={"FP32_existing_135M":{},"CB186":{},"GT"+str(new):{}}
    cases=[]
    for i in range(16):
        cases.append({"case_id":i,"question":str(i),"gold":"x","domain":"arithmetic"})
        for name in statuses:
            statuses[name][i]="machine_correct" if i<({"FP32_existing_135M":6,"CB186":2,"GT200":3,"GT204":2}.get(name,0)) else "ungraded"
    scores={}
    for name,s in statuses.items():
        rows=[{"case_id":i,"domain":"arithmetic","machine_status":s[i],"raw":f"{name}-{i}"} for i in range(16)]
        scores[name]={"n":16,"machine_confirmed":sum(v=="machine_correct" for v in s.values()),
        "machine_recognized_wrong":0,"ungraded_not_known_wrong":sum(v=="ungraded" for v in s.values()),
        "verbatim_answers_and_grades":rows}
    # original and starting v186 exact rows MUST be identical across reports
    for name in ("FP32_existing_135M","CB186"):
        for row in scores[name]["verbatim_answers_and_grades"]:
            row["raw"]=f"immutable-{name}-{row['case_id']}"
    return {"cases":cases,"preexisting_v192_frozen_grader_sha256":"frozen-grader","scores":scores}


def train():
    steps=[{"step":i,"operator":f"layer.{i}","source_step_accepted":i<13} for i in range(1,17)]
    return {"steps":steps,"actual_pilot_steps":16,
       "accepted_steps_before_full_TRAIN_guard":12,
       "all_pilot_mutations_rolled_back_on_full_TRAIN_fail":False}


def summary(n):
    return {"completed":True,"source_SHA256":str(n)*64,
      "whole_original_source_bpp":0.6678394131307638,
      "native_CPU_speed_ratio_CB59":0.97}


class TestReadOnlyForensic(unittest.TestCase):
    def test_strict_frozen_comparison(self):
        r=analyze(qa(200),qa(204),train(),train(),summary(2),summary(4))
        self.assertEqual((r["original_FP32_machine_correct"],
                         r["v200_machine_correct"],r["v204_machine_correct"]), (6,3,2))
        self.assertEqual(r["v200_v204_status_transitions"]["machine_correct -> ungraded"],1)
        self.assertTrue(r["public_DEV_must_not_be_used_as_optimizer_or_checkpoint_selector"])

    def test_gold_drift_hard_fail(self):
        a,b=qa(200),qa(204)
        b["cases"][0]["gold"]="altered"
        with self.assertRaises(AssertionError):analyze(a,b,train(),train(),summary(2),summary(4))

    def test_fp32_reference_drift_hard_fail(self):
        a,b=qa(200),qa(204)
        b["scores"]["FP32_existing_135M"]["verbatim_answers_and_grades"][0]["raw"]="altered"
        with self.assertRaises(AssertionError):analyze(a,b,train(),train(),summary(2),summary(4))

    def test_grader_drift_hard_fail(self):
        a,b=qa(200),qa(204)
        b["preexisting_v192_frozen_grader_sha256"]="changed"
        with self.assertRaises(AssertionError):analyze(a,b,train(),train(),summary(2),summary(4))

    def test_step_schedule_drift_hard_fail(self):
        a,b=train(),train()
        b["steps"][0]["operator"]="different"
        with self.assertRaises(AssertionError):analyze(qa(200),qa(204),a,b,summary(2),summary(4))


if __name__=="__main__":
    unittest.main()
