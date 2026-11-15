"""S13 physical success cannot be misread as qualifying new language ability."""
from __future__ import annotations
import unittest
from research.nra3_s13.research_acceptance_audit import audit


def _report():
    return {
        "stage":"NRA3_S13_REAL_43M_S10_TRAIN72_1266_PAID_FP16_CAUSAL_RELATION_LINGUISTIC_READOUT_SHADOW",
        "S13_real_calibri43M_train72_rows":72,
        "S13_train_only_words":408,
        "S13_new_rank8_feedback_unique_parameters":1266,
        "S13_new_rank8_feedback_physical_bytes":2581,
        "S13_heldout16_S12c_now_CONSUMED_and_not_loaded":True,
        "S13_native_all_43_matrices_still_physically_packed_inference_C":True,
        "no_LIVE_Render_Supabase_APK_Jessica_chat":True,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
        "S13_any_independent_new_eval_or_full_answer_EOS_measured":False,
        "S13_raw_area_plus_selector_plus_feedback_bpp":.859934712619438,
        "S13_original_direct_causal_UPOS_train_accuracy":{"correct":245,"total":408},
        "S13_feedback_direct_causal_UPOS_train_accuracy":{"correct":208,"total":408},
        "S13_original_direct_causal_role_train_accuracy":{"correct":247,"total":408},
        "S13_feedback_direct_causal_role_train_accuracy":{"correct":249,"total":408},
        "S13_original_known_past_head_train_accuracy":{"correct":147,"total":192},
        "S13_feedback_known_past_head_train_accuracy":{"correct":147,"total":192}
    }


class S13HardRejection(unittest.TestCase):
    def test_actual_observed_UPOS_hard_accuracy_regression_fails_closed(self):
        r=audit(_report())
        self.assertFalse(r["research_acceptance"])
        self.assertFalse(r["activation_allowed"])
        self.assertEqual(r["output_gate"],0)
        self.assertTrue(r["candidate_regressed_UPOS"])
        self.assertEqual(r["comparison"]["UPOS"]["delta_correct"],-37)
        self.assertEqual(r["comparison"]["semantic_roles"]["delta_correct"],2)
        self.assertEqual(r["comparison"]["past_only_syntax_head"]["delta_correct"],0)
        self.assertIn("hard_causal_UPOS_accuracy_REGRESSION_on_train72",r["reasons"])
        self.assertFalse(r["S12c_consumed_eval16_touched_for_this_audit"])

    def test_even_better_training_does_not_promote_without_independent_EVAL(self):
        x=_report()
        x["S13_feedback_direct_causal_UPOS_train_accuracy"]["correct"]=300
        x["S13_feedback_direct_causal_role_train_accuracy"]["correct"]=280
        x["S13_feedback_known_past_head_train_accuracy"]["correct"]=160
        r=audit(x)
        self.assertFalse(r["research_acceptance"])
        self.assertFalse(r["candidate_regressed_UPOS"])
        self.assertFalse(r["candidate_accepted"])
        self.assertIn("independent_NEW_Russian_EVAL_after_S13_lock_not_run",r["reasons"])

    def test_reject_gate_flip_and_claimed_independent_evaluation(self):
        for field,new in (
            ("output_gate",.1),("candidate_accepted",True),
            ("activation_allowed",True),
            ("S13_any_independent_new_eval_or_full_answer_EOS_measured",True),
            ("S13_heldout16_S12c_now_CONSUMED_and_not_loaded",False),
            ("S13_native_all_43_matrices_still_physically_packed_inference_C",False),
        ):
            x=_report();x[field]=new
            with self.assertRaises(ValueError):
                audit(x)


if __name__=="__main__":
    unittest.main()
