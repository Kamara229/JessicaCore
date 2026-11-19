"""S71 old TRAIN32 exact native J7 source comparison contracts, no model."""
import unittest

from research.nra3_s20.s71_S132_allstep_phase_original_oldtrain32_J7 import (
    actual_original_s190_frozen_source_probes,final_audit,sha,
)


class TestS71OriginalS132AllStepJ7(unittest.TestCase):
    def test_existing_source_only_old_train32_probes(self):
        p=actual_original_s190_frozen_source_probes()
        self.assertEqual(len(p),32)
        self.assertEqual(len({sha(q) for q,_,_ in p}),32)

    def test_no_claim_of_full_616_or_streaming_causal_safety(self):
        def row(i):
            return {
                "source_sha256":f"{i:064x}",
                "source_group":"old_32_training",
                "output_gate":0,"candidate_accepted":False,
                "original_source_only_no_answer_used_for_causal_phase":True,
                "exact_original_J7_token_match":i%3!=0,
                "first_token_same":i%5!=0,
                "candidate_last_token_is_original_EOS":i%7!=0,
                "incremental_decode_calls":2,
            }
        rows=[row(i) for i in range(32)]
        audit=final_audit(rows,64)
        self.assertEqual(audit["n_previous_TRAIN32"],32)
        self.assertEqual(audit["real_original_decoding_subcalls_incremental_total"],64)
        self.assertFalse(audit["not_616_full_new_candidate_retention_certified"] is False)
        self.assertFalse(audit["S132_allstep_phase_production_safety_proven"])
        self.assertFalse(audit["streaming_causal_safety_certified"])
        self.assertEqual(audit["output_gate"],0)
        rows[0]["candidate_accepted"]=True
        with self.assertRaisesRegex(ValueError,"old32"):
            final_audit(rows,64)

    def test_fail_closed_incomplete_training_source(self):
        with self.assertRaisesRegex(ValueError,"old32"):
            final_audit([],0)

if __name__=="__main__":
    unittest.main()
