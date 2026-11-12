"""S181 offline replay unit contracts, mock tokenizer. No model/Jessica test."""
from pathlib import Path
import unittest

from calibri2.s181.full_456_tokenized_shadow_replay import (
    counterfactual_offline, hashed,
)
from calibri2.s181.source_128_frozen_456_posthoc import reconstruct_sources


class FakeTokenizer:
    eos_token_id = 999
    def encode(self, phrase, add_special_tokens=False):
        assert not add_special_tokens
        tokens = phrase.split()
        if len(tokens) != 2:
            return [7]
        return [100 + sum(map(ord, tokens[0])), 200 + sum(map(ord, tokens[1]))]


def case_data():
    root = Path(__file__).resolve().parents[3]
    source = reconstruct_sources(root)
    tok = FakeTokenizer()
    questions = {}
    gold = []
    traces = []
    for i, (key, question, answer) in enumerate(source):
        assert key not in questions
        questions[key] = question
        h = hashed(question)
        expected = tok.encode(answer) + [tok.eos_token_id]
        old = [17, 18, tok.eos_token_id] if i < 26 else expected[:]
        gold.append({"id": key, "question_sha256": h, "target_ids": expected})
        traces.append({"prompt_sha256": h, "full_generated_ids": old,
                       "gold_seen_at_inference": False})
    for i in range(328):
        key, question = f"OTHER/{i}", f"OTHER authorized independent TRAIN {i}"
        questions[key] = question
        h = hashed(question)
        expected = [10, 20, tok.eos_token_id]
        old = [21, 22, tok.eos_token_id] if i < 2 else expected[:]
        gold.append({"id": key, "question_sha256": h, "target_ids": expected})
        traces.append({"prompt_sha256": h, "full_generated_ids": old,
                       "gold_seen_at_inference": False})
    assert len(questions) == len(gold) == len(traces) == 456
    trace = {"stage": "S179_TRACE_ONLY", "number": 456,
             "labels_used": False, "cases": traces}
    s87 = {"stage": "S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT",
           "all_456_per_identity_results": gold}
    return questions, tok, trace, s87, source


class S181Full456OverlayContracts(unittest.TestCase):
    def test_offline_456_source_only_no_forgotten(self):
        questions, tok, trace, s87, source = case_data()
        r = counterfactual_offline(questions, tok, trace, s87, source)
        x = r["metrics"]
        self.assertEqual(x["total"], 456)
        self.assertEqual(x["shadow_matched"], 128)
        self.assertEqual(x["shadow_abstained"], 328)
        self.assertEqual(x["before_full_correct"], 428)
        self.assertEqual(x["offline_after_full_correct"], 454)
        self.assertEqual(x["gained_full"], 26)
        self.assertEqual(x["lost_full"], 0)
        self.assertEqual(x["lost_first"], 0)
        self.assertTrue(r["not_actual_integrated_CPU_J7"])
        self.assertTrue(r["not_generalization_proof_due_to_train_template"])
        self.assertTrue(r["cpu_speed_unmeasured"])
        self.assertTrue(r["physical_bpp_unmeasured"])
        self.assertTrue(r["not_model_accepted"])
        self.assertNotIn("expected", str(r))
        self.assertNotIn("target_ids", str(r))

    def test_source_prompt_identity_drift_fails_closed(self):
        questions, tok, trace, s87, source = case_data()
        key = source[0][0]
        questions[key] = "TRAIN item 5: malicious replacement"
        with self.assertRaisesRegex(AssertionError, "literal question ABI"):
            counterfactual_offline(questions, tok, trace, s87, source)

    def test_non_source_matches_are_rejected(self):
        questions, tok, trace, s87, source = case_data()
        p = source[0][1]
        key = "OTHER/12"
        questions[key] = p + " "
        with self.assertRaises(AssertionError):
            counterfactual_offline(questions, tok, trace, s87, source)

    def test_original_baseline_drift_fails_closed(self):
        questions, tok, trace, s87, source = case_data()
        trace["cases"][-1]["full_generated_ids"] = [17, 18, 999]
        with self.assertRaisesRegex(AssertionError, "baseline drift"):
            counterfactual_offline(questions, tok, trace, s87, source)


if __name__ == "__main__":
    unittest.main()
