"""S70 frozen native-prefix instrumentation ABI: synthetic, no weights/QA."""
import unittest
from types import SimpleNamespace

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s70_true43m_native_s95_source_prefix_counterfactual import (
    source_prefix_from_safe_cut,audit_rows,
)


class TestS70OriginalNativeS95(unittest.TestCase):
    def test_original_bpe_native_boundary_only(self):
        s="Диспетчер передал врачу документы, но сказанное было цитатой."
        words=[
            SimpleNamespace(index=i,causally_safe=True,
                            shared_native_tokens=(),
                            native_token_positions=(4,5) if i==1 else (2,3))
            for i,_ in enumerate(raw_surface_grammar(s))
        ]
        got=source_prefix_from_safe_cut(s,words,1,list(range(17)),6)
        self.assertEqual(got,s[:raw_surface_grammar(s)[1].end])
        words[1].shared_native_tokens=(5,)
        with self.assertRaisesRegex(ValueError,"original BPE-safe"):
            source_prefix_from_safe_cut(s,words,1,list(range(17)),6)

    def test_source_position_must_match_native_cut_not_guess(self):
        s="Сначала были бумаги, затем подписали акт."
        w=[SimpleNamespace(index=i,causally_safe=True,
                            shared_native_tokens=(),native_token_positions=(3,4))
           for i,_ in enumerate(raw_surface_grammar(s))]
        with self.assertRaisesRegex(ValueError,"original BPE-safe"):
            source_prefix_from_safe_cut(s,w,0,list(range(17)),6)

    def test_eight_rows_gate_off_and_no_online_causal_certification(self):
        def r():
            z={"max_abs":0.04}
            return {"original_native_ids_prefix_reused_bit_identical":True,
                    "true_original_native_BPE_not_byte_surrogate":True,
                    "normal_generation_safety_or_root_cause_proven":False,
                    "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
                    "features_original_full_vs_prefix":z,
                    "features_prefix_vs_new_suffix":z,
                    "same_native_prefix_H8_full_vs_prefix":z,
                    "same_native_prefix_H8_prefix_vs_suffix":z,
                    "same_native_prefix_logits_full_vs_prefix":z,
                    "same_native_prefix_logits_prefix_vs_suffix":z}
        data=[r() for _ in range(8)]
        report=audit_rows(data)
        self.assertEqual(report["source_feature_full_vs_prefix_changed"],8)
        self.assertEqual(report["same_native_H8_full_vs_prefix_changed"],8)
        self.assertFalse(report["online_causal_safety_certified"])
        with self.assertRaisesRegex(ValueError,"eight"):
            audit_rows(data[:7])
        data[3]["candidate_accepted"]=True
        with self.assertRaisesRegex(ValueError,"candidate/online"):
            audit_rows(data)


if __name__=="__main__":
    unittest.main()
