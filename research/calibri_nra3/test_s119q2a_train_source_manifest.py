"""S119q2a contract tests: ALL author-created synthetic, no model, no GOLD."""
import copy
import hashlib
import json
import unittest
from collections import Counter

from research.calibri_nra3.s119q2a_train_source_manifest import (
    VERSION, KINDS, SPLITS, SPENT_S119N_PROMPTS, EXPECTED_COUNTS,\n    EXPECTED_MANIFEST_SHA256,
    TRAIN_MALE, CAL_MALE, TRAIN_FEMALE, CAL_FEMALE, authored_row,
    build_rows, validate_rows, build_manifest, canonical_bytes, sha,
)


class S119Q2AAuthorCreatedTrainingSource(unittest.TestCase):
    def setUp(self):
        self.rows=build_rows()

    def test_01_exact_rows_counts_and_all_six_task_buckets(self):
        d=validate_rows(self.rows)
        self.assertEqual(d["rows_total"],90)
        self.assertEqual(d["split_counts"],EXPECTED_COUNTS)
        self.assertEqual(Counter(x["split"] for x in self.rows),
                         EXPECTED_COUNTS)
        self.assertEqual(Counter((x["split"],x["kind"]) for x in self.rows),
                         {("TRAIN",k):12 for k in KINDS}|
                         {("CALIBRATION",k):3 for k in KINDS})
        self.assertTrue(d["no_final_blind_test_created"])

    def test_02_known_grounded_grammar_and_arithmetic_rules(self):
        a=authored_row("roles_word_order","TRAIN",0)
        self.assertIn("Артём передал книгу Диане",a["prompt"])
        self.assertEqual(a["completion"],"Диана.")
        b=authored_row("roles_word_order","TRAIN",1)
        self.assertEqual(b["completion"],"Егор.")
        self.assertTrue(b["prompt"].startswith("Папку"))
        n=authored_row("negation","TRAIN",0)
        self.assertIn(" не передал ",n["prompt"])
        self.assertEqual(n["completion"],"Нет.")
        y=authored_row("negation","TRAIN",1)
        self.assertEqual(y["completion"],"Да.")
        h=authored_row("transitive_comparison","TRAIN",2)
        self.assertEqual(h["completion"],"Никита.")
        self.assertEqual(authored_row(
            "arithmetic","TRAIN",0)["completion"],"7.")
        self.assertEqual(authored_row(
            "arithmetic","TRAIN",1)["completion"],"10.")

    def test_03_calibration_names_distinct_and_no_spent_prompts(self):
        train_names={n[0] for n in (*TRAIN_MALE,*TRAIN_FEMALE)}
        cal_names={n[0] for n in (*CAL_MALE,*CAL_FEMALE)}
        self.assertFalse(train_names & cal_names)
        self.assertFalse(set(SPENT_S119N_PROMPTS) &
                         {x["prompt"] for x in self.rows})
        self.assertTrue(all(x["source_kind"].startswith("original_")
                            for x in self.rows))

    def test_04_manifest_sha_and_sorted_utf8_bytes_reproducible(self):
        first=build_manifest(self.rows)
        second=build_manifest(build_rows())
        self.assertEqual(first,second)
        self.assertEqual(len(first["synthetic_rows_SHA256"]),64)\n        self.assertEqual(first["synthetic_rows_SHA256"],EXPECTED_MANIFEST_SHA256)
        self.assertEqual(first["synthetic_rows_SHA256"],
                         sha(b"".join(canonical_bytes(r) for r in self.rows)))
        self.assertEqual(first["split_SHA256"]["TRAIN"],
                         sha(b"".join(canonical_bytes(r) for r in self.rows
                                         if r["split"]=="TRAIN")))
        self.assertNotEqual(first["split_SHA256"]["TRAIN"],
                            first["split_SHA256"]["CALIBRATION"])
        self.assertNotIn("\\u0410",canonical_bytes(self.rows[0]).decode("utf8"))

    def test_05_reject_modified_expected_label_even_valid_russian(self):
        rows=copy.deepcopy(self.rows)
        rows[0]["completion"]="Некоторое другое имя."
        with self.assertRaisesRegex(ValueError,"generator output"):
            validate_rows(rows)

    def test_06_reject_spoofed_splitting(self):
        rows=copy.deepcopy(self.rows)
        rows[0]["split"]="CALIBRATION"
        with self.assertRaisesRegex(ValueError,"generator output"):
            validate_rows(rows)

    def test_07_reject_any_missing_or_extra_sample(self):
        with self.assertRaisesRegex(ValueError,"exactly 90"):
            validate_rows(self.rows[:-1])
        with self.assertRaisesRegex(ValueError,"exactly 90"):
            validate_rows(self.rows+[self.rows[0]])

    def test_08_reject_spent_original_S119n_prefix(self):
        rows=copy.deepcopy(self.rows)
        rows[0]["prompt"]=SPENT_S119N_PROMPTS[0]
        with self.assertRaisesRegex(ValueError,"generator output"):
            validate_rows(rows)

    def test_09_reject_license_or_provenance_forgery(self):
        rows=copy.deepcopy(self.rows)
        rows[0]["rights_scope"]="external_unverified_corpus"
        with self.assertRaisesRegex(ValueError,"generator output"):
            validate_rows(rows)

    def test_10_readonly_calibration_and_no_final_eval(self):
        manifest=build_manifest(self.rows)
        self.assertTrue(manifest["not_frozen_final_independent_eval"])
        self.assertFalse(manifest["candidate_accepted"])
        self.assertEqual(manifest["output_gate"],0)
        self.assertTrue(manifest["metrics"]["no_training_or_gradients"])
        self.assertTrue(manifest["metrics"]["no_model_or_tokenizer_loaded"])
        self.assertTrue(manifest["metrics"]["third_party_source_loaded"] is False)

    def test_11_invalid_rows_are_fail_closed(self):
        for kind,split,index in (
            ("illicit","TRAIN",0),("roles_word_order","EVAL",0),
            ("event_time","CALIBRATION",4),("arithmetic","TRAIN",-1),
        ):
            with self.subTest(kind=kind,split=split,index=index):
                with self.assertRaisesRegex(ValueError,"predeclared"):
                    authored_row(kind,split,index)

    def test_12_truncated_or_corrupted_dataset_identity(self):
        rows=copy.deepcopy(self.rows)
        rows[0]["id"]=rows[1]["id"]
        with self.assertRaisesRegex(ValueError,"generator output"):
            validate_rows(rows)

    def test_13_cross_split_prompt_list_has_no_exact_repetition(self):
        train={x["prompt"] for x in self.rows if x["split"]=="TRAIN"}
        cal={x["prompt"] for x in self.rows if x["split"]=="CALIBRATION"}
        self.assertEqual(len(train),72)
        self.assertEqual(len(cal),18)
        self.assertFalse(train & cal)

    def test_14_no_json_lines_contain_the_earlier_S119n_six_prompts(self):
        blob=b"".join(canonical_bytes(x) for x in self.rows)
        for old_prompt in SPENT_S119N_PROMPTS:
            self.assertNotIn(old_prompt.encode("utf8"),blob)
        self.assertEqual(len(blob.splitlines()),90)

    def test_15_calibration_is_not_an_independent_human_quality_benchmark(self):
        d=build_manifest(self.rows)
        self.assertTrue(d["metrics"]["only_template_synthetic"])
        self.assertTrue(d["metrics"]["no_independent_quality_claim"])
        self.assertTrue(d["metrics"]["no_protected_original_616_J7_or_S118_B_GOLD_used"])
        self.assertTrue(d["metrics"]["no_spent_EVAL128_or_S119n_prompt_used"])


if __name__=="__main__":
    unittest.main()
