"""S182: selector source-only negative contracts. No model import/test here."""
import unittest

from calibri2.s182.real_cpu_j7_shadow_guard import candidate_only, STATS
from calibri2.s181.source_grammar_shadow import S30_TEMPLATES, S54_TEMPLATES


class MockTok:
    eos_token_id = 2
    def encode(self, phrase, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("S182 no chat gold")
        return {"red bowl": [10, 20], "red drum": [10, 21],
                "green bowl": [11, 22], "green rope": [11, 23]}.get(phrase, [2])


class S182SourceOnlySafety(unittest.TestCase):
    def test_exact_S30_and_S54_source_pair_admitted(self):
        tok = MockTok()
        for templ in (*S30_TEMPLATES, *S54_TEMPLATES):
            q = templ.format(i=9000,a="OwnerA",b="OwnerB",left="red bowl",right="red drum")
            self.assertEqual(candidate_only(q,tok,[[10,20],[10,21]]),[10,20])

    def test_nonexact_source_and_wrong_source_span_abstain(self):
        tok = MockTok()
        t = S30_TEMPLATES[0]
        q = t.format(i=0,a="OwnerA",b="OwnerB",left="red bowl",right="red drum")
        self.assertIsNone(candidate_only(q,tok,[[10,21]]))
        self.assertIsNone(candidate_only(q+" extra",tok,[[10,20],[10,21]]))
        self.assertIsNone(candidate_only("TRAIN item 0: negative owner claim",tok,[[10,20]]))

    def test_wrong_len_and_eos_rejected(self):
        tok=MockTok()
        t=S54_TEMPLATES[0]
        q=t.format(i=0,a="OwnerA",b="OwnerB",left="blue ring",right="blue rope")
        self.assertIsNone(candidate_only(q,tok,[[2],[3]]))
        self.assertTrue(STATS["invalid_tokens_rejected"] >= 1)

    def test_early_legacy_source_fallback_no_gold_fields(self):
        self.assertEqual(candidate_only.__code__.co_argcount,3)
        self.assertNotIn("expected",candidate_only.__code__.co_varnames)
        self.assertNotIn("gold",candidate_only.__code__.co_varnames)


if __name__ == "__main__":
    unittest.main()
