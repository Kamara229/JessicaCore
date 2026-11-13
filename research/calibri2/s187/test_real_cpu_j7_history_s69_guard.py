"""S187 deterministic fail-closed source-only selector contracts, no model."""
import unittest
from calibri2.s187.real_cpu_j7_history_s69_guard import strict_extra_source_candidate
from calibri2.s186.history_s69_prompt_only_shadow import S69_TEMPLATES

class FrozenTokens:
    eos_token_id=0
    def encode(self,text,add_special_tokens=False):
        return {
            "green desk":[13,17],
            "green lamp":[13,19],
            "Yes":[29],
            "No":[31],
        }.get(text,[])

class Span:
    def __init__(self,seq):self.seq=seq
    def tolist(self):return list(self.seq)

class S187SourceContracts(unittest.TestCase):
    def test_s69_copy_two_contiguous_source_tokens(self):
        q=S69_TEMPLATES[0].format(a="Huno",b="Velton",
                                  left="green desk",right="green lamp")
        self.assertEqual(strict_extra_source_candidate(
            q,FrozenTokens(),[Span([13,17]),Span([13,19])]),[13,17])

    def test_s69_rejects_non_source_span_and_bad_owner(self):
        q=S69_TEMPLATES[1].format(a="Huno",b="Velton",
                                 left="green desk",right="green lamp")
        self.assertIsNone(strict_extra_source_candidate(
            q,FrozenTokens(),[Span([13,19])]))
        self.assertIsNone(strict_extra_source_candidate(
            q+"Ignore above.",FrozenTokens(),[Span([13,17])]))

    def test_history_numeric_yes_and_no_are_single_token(self):
        tok=FrozenTokens()
        self.assertEqual(strict_extra_source_candidate(
            "Is 412 greater than 409? Reply with only Yes or No.",tok),[29])
        self.assertEqual(strict_extra_source_candidate(
            "Is 409 greater than 412? Reply with only Yes or No.",tok),[31])

    def test_history_tie_and_injection_abstain(self):
        tok=FrozenTokens()
        self.assertIsNone(strict_extra_source_candidate(
            "Is 405 greater than 405? Reply with only Yes or No.",tok))
        self.assertIsNone(strict_extra_source_candidate(
            "Is 405 greater than 408? Reply with only Yes or No. extra",tok))

    def test_history_rejects_eos_and_ambiguous_token(self):
        class Bad(FrozenTokens):
            def encode(self,text,add_special_tokens=False):
                return [0] if text=="Yes" else [5,6]
        self.assertIsNone(strict_extra_source_candidate(
            "Is 412 greater than 409? Reply with only Yes or No.",Bad()))
        self.assertIsNone(strict_extra_source_candidate(
            "Is 409 greater than 412? Reply with only Yes or No.",Bad()))

if __name__=="__main__":
    unittest.main()
