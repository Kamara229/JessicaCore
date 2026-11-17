"""S25 source-only provenance and raw negative scope semantics, not CPU tests."""
from __future__ import annotations
import inspect
import unittest
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar,FEATURES
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    build_s22_train256,PATTERNS,
)
from research.nra3_s20.s25_real43m_rejected_raw_cue_alignment_audit import cue_audit


class S25GenuineCPUInputCueContracts(unittest.TestCase):
    def test_256_raw_text_8_rejected_preposed_has_negative_dative_word(self):
        cases=build_s22_train256()
        pat=25
        self.assertEqual(PATTERNS[pat],"rejected_recipient_preposed")
        for i in range(8):
            g=cases[32*i+pat]
            neg_args=[a for a in g.arguments if a.polarity=="REJECTED"]
            self.assertEqual(len(neg_args),1)
            rejected=neg_args[0]
            token=raw_surface_grammar(g.text)[rejected.token-1]
            f=token.features
            self.assertEqual(f[FEATURES.index("previous_word_negative_particle")],1)
            self.assertEqual(f[FEATURES.index("ending_dative_possible")],1)
            pred=next(e.predicate for e in g.events if e.event_id==rejected.event_id)
            self.assertLess(rejected.token,pred)

    def test_raw_input_audit_does_not_fit_or_forward_neural_model(self):
        s=inspect.getsource(cue_audit)
        for absent in ("forward(", "optimizer(", "torch.optim",
                       "backward(", "inference_mode(",
                       "load_state_dict", "eval("):
            self.assertNotIn(absent,s)
        self.assertIn("surface_previous_word_negative_particle",s)
        self.assertIn("surface_likely_dative_suffix",s)
        self.assertIn("gold label is not REJECTED recipient",s)


if __name__=="__main__":
    unittest.main()
