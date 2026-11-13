"""S3J offset-alignment tests: no 43M, no gold, no LIVE."""
import unittest
from research.nra2_bridge.stage3j_token_entity_alignment import (
    node_mention_alignment,validate_without_gold,MENTION,
)
from research.nra2_bridge.stage3g_factorized_graph_controls import build_rows


class CharTokenizer:
    """A deterministic fake *offset-aware* tokenizer for ABI testing."""
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
        assert len(messages)==1 and add_generation_prompt and not tokenize
        return "<|R5_role|> "+messages[0]["content"]+" <|assistant|>"

    def encode(self,s,add_special_tokens=False):
        assert not add_special_tokens
        return [ord(c) for c in s]

    def __call__(self,s,add_special_tokens=False,return_offsets_mapping=False):
        assert not add_special_tokens and return_offsets_mapping
        return {"input_ids":self.encode(s),
                "offset_mapping":[(i,i+1) for i in range(len(s))]}


class DriftingTokenizer(CharTokenizer):
    def __call__(self,s,add_special_tokens=False,return_offsets_mapping=False):
        d=super().__call__(s,add_special_tokens,return_offsets_mapping)
        d["input_ids"][0]=-1
        return d


class NoOffsetTokenizer(CharTokenizer):
    def __call__(self,s,add_special_tokens=False,return_offsets_mapping=False):
        raise NotImplementedError("slow tokenizers without offsets fail closed")


class AmbiguousMergedTokenizer(CharTokenizer):
    def __call__(self,s,add_special_tokens=False,return_offsets_mapping=False):
        d=super().__call__(s,add_special_tokens,return_offsets_mapping)
        a=s.index("R0 -> R1")
        # Two text identities both covered by one artificial (bad) span
        # with an otherwise exact ID list. Must reject this ambiguity.
        d["offset_mapping"][a]=(a,a+8)
        return d


class Stage3JTests(unittest.TestCase):
    def test_all128_real_research_questions_without_gold(self):
        rows=build_rows()
        ret=validate_without_gold(CharTokenizer(),[r.question for r in rows])
        self.assertEqual(len(ret),128)
        for row,aligned in zip(rows,ret):
            self.assertEqual(row.prompt_sha256,aligned["prompt_sha256"])
            self.assertEqual(len(aligned["mention_mask"]),aligned["tokens"])
            self.assertTrue(all(sum(m)<=1 for m in aligned["mention_mask"]))
            self.assertGreaterEqual(len(aligned["mentions"]),8)
            self.assertIn(row.source,aligned["node_identities_present"])
            self.assertIn(row.target,aligned["node_identities_present"])
            self.assertNotIn("adjacency",aligned)
            self.assertNotIn("reachable",aligned)
            self.assertNotIn("answer",aligned)
            self.assertNotIn("source",aligned)
            self.assertNotIn("target",aligned)

    def test_role_sentinel_not_counted_as_node(self):
        z=node_mention_alignment(CharTokenizer(),"Follow R0 -> R1. Can R1 reach R0?")
        self.assertEqual(z["node_identities_present"],[0,1])
        self.assertEqual([m.node for m in z["mentions"]],[0,1,1,0])
        self.assertNotIn(5,z["node_identities_present"])

    def test_do_not_match_partial_identifiers(self):
        z=node_mention_alignment(CharTokenizer(),"R0 is next to R1; not R12, XR3, R0_extra.")
        self.assertEqual([m.node for m in z["mentions"]],[0,1])

    def test_exact_native_prefill_id_guard(self):
        tok=CharTokenizer()
        question="A: R1 -> R2. Is R2 after R1?"
        prompt=tok.apply_chat_template([{"role":"user","content":question}],
                                       tokenize=False,add_generation_prompt=True)
        ids=tok.encode(prompt,add_special_tokens=False)
        self.assertEqual(node_mention_alignment(tok,question,ids)["input_ids"],ids)
        with self.assertRaisesRegex(AssertionError,"prefill ids drift"):
            node_mention_alignment(tok,question,ids[:-1])

    def test_missing_offsets_or_tokenizer_drift_fail_closed(self):
        question="R0 -> R1. Does R0 reach R1?"
        with self.assertRaisesRegex(AssertionError,"offset mappings"):
            node_mention_alignment(NoOffsetTokenizer(),question)
        with self.assertRaisesRegex(AssertionError,"offset ABI drift"):
            node_mention_alignment(DriftingTokenizer(),question)

    def test_multi_node_token_fails_closed(self):
        question="R0 -> R1. Does R0 reach R1?"
        with self.assertRaisesRegex(AssertionError,"merges distinct node"):
            node_mention_alignment(AmbiguousMergedTokenizer(),question)

    def test_duplicate_prompt_fail_closed(self):
        q="R0 -> R1"
        with self.assertRaisesRegex(AssertionError,"distinct"):
            validate_without_gold(CharTokenizer(),[q,q])


if __name__=="__main__":
    unittest.main()
