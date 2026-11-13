"""S185 456-prompt preflight and SHA-only posthoc fail-closed contracts."""
import copy
import unittest
from calibri2.s185.first_eos_source_only_feasibility import (
    classify_after_prompt_only_proposals,
)


def fixture():
    proposals={}
    for i in range(456):
        sha=f"{i:064x}"
        proposals[sha]={
            "source_family": ("S82_new" if i%2 else "S30"),
            "source_identity":f"S82_new/{i}/left",
            "original_full_length":(2 if i==455 else 3),
            "original_terminates_in_EOS":i!=455,
            "prompt_only_copy_proposal_admissible":i==454,
            "unchanged_model_first_token_vs_source_proposal":"different" if i==454 else "not-applicable",
            "strict_source_span_two_token_EOS_append_admissible":i==455,
        }
    post={
        "event":"S184_LAST_TWO_SOURCE456_FROZEN_J7_POSTHOC_SHA_ONLY",
        "remaining_incorrect":2,"model_accepted":False,
        "no_labels_or_answers_enter_model_or_selector":True,
        "remaining_cases_SHA_only":[
            {"prompt_sha256":f"{454:064x}","failure_family_posthoc":"first-token","original_preexisting_guard_events":[]},
            {"prompt_sha256":f"{455:064x}","failure_family_posthoc":"EOS-or-length","original_preexisting_guard_events":[]},
        ]
    }
    trace={"stage":"S179_TRACE_ONLY","labels_used":False,"number":456}
    return proposals,post,trace


class S185OfflineContracts(unittest.TestCase):
    def test_first_and_eos_policy_classifies_before_posthoc(self):
        report=classify_after_prompt_only_proposals(*fixture())
        self.assertEqual(sum(report["all_456_source_families"].values()),456)
        self.assertEqual(len(report["two_residual_source_diagnostics_SHA_only"]),2)
        self.assertFalse(report["candidate_selection_by_gold_or_SHA"])
        self.assertTrue(report["requires_new_REAL_J7_456plus128_before_any_acceptance"])
        self.assertFalse(report["model_accepted"])
        self.assertNotIn("target_ids",str(report))
        self.assertNotIn("full_generated_ids",str(report))

    def test_no_source_456_preclassification_fails(self):
        a=list(fixture())
        a[0].pop(f"{0:064x}")
        with self.assertRaises(AssertionError):
            classify_after_prompt_only_proposals(*a)

    def test_posthoc_provenance_fails_closed(self):
        a=list(fixture())
        a[1]["remaining_incorrect"]=1
        with self.assertRaises(AssertionError):
            classify_after_prompt_only_proposals(*a)

    def test_posthoc_missing_sha_fails_closed(self):
        a=list(fixture())
        a[1]["remaining_cases_SHA_only"][0]["prompt_sha256"]="z"*64
        with self.assertRaises(AssertionError):
            classify_after_prompt_only_proposals(*a)

    def test_label_marked_trace_fails_closed(self):
        a=list(fixture())
        a[2]["labels_used"]=True
        with self.assertRaises(AssertionError):
            classify_after_prompt_only_proposals(*a)


if __name__=="__main__":
    unittest.main()
