"""S74 source/target wall and contrastive scenario first-TRAIN-only tests."""
import ast
import unittest
from collections import Counter
from pathlib import Path

from research.nra3_s20.s74_newtrain120_source_ONLY import (
    CASES_PER_FAMILY,FAMILIES,GROUP_TRAIN,GROUP_DIAG,
    new_source_only_train120,source_sha256,grouped_train_source_only,
)
from research.nra3_s20.s74_newtrain120_gold_CONSTRUCTION_ONLY_never_in_forward import (
    CONTRASTS,AREAS,constructed_NEW_TRAIN_gold120_only_for_loss,
    new_train120_source_gold_alignment_manifest,
)


class TestS74NewSituationTrainOnly(unittest.TestCase):
    def test_source_gold_isolation_with_no_label_fields(self):
        x=new_source_only_train120()
        self.assertEqual(len(x),120)
        self.assertEqual(len(FAMILIES),12)
        self.assertEqual(CASES_PER_FAMILY,10)
        self.assertTrue(all(set(q)=={"row_id","family","text"} for q in x))
        self.assertEqual(len({q["text"] for q in x}),120)
        self.assertEqual(len({q["row_id"] for q in x}),120)
        self.assertTrue(all(q["text"].strip()==q["text"] for q in x))
        self.assertEqual(set(Counter(z["family"] for z in x).values()),{10})

    def test_new_train_gold_is_only_separate_loss_input(self):
        raw=new_source_only_train120()
        gold=constructed_NEW_TRAIN_gold120_only_for_loss()
        self.assertEqual(len(gold),len(raw))
        self.assertEqual([x["row_id"] for x in gold],
                         [x["row_id"] for x in raw])
        self.assertTrue(all(x["TRAIN_only_never_in_source_features"] for x in gold))
        self.assertTrue(all(not x["human_independent_gold_verified"] for x in gold))
        self.assertTrue(all(x["source_sha256_before_any_native_forward"]
                            ==source_sha256() for x in gold))

    def test_s72_area_names_schema_order_matches_labels(self):
        p=Path("research/nra3_s20/s72_nra2_situation_microareas.py")
        tree=ast.parse(p.read_text(encoding="utf8"))
        declared=[]
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(
                isinstance(t,ast.Name) and t.id=="AREAS" for t in node.targets):
                declared=list(ast.literal_eval(node.value))
        self.assertEqual(declared,list(AREAS))
        self.assertEqual(len(AREAS),8)
        self.assertEqual(len(CONTRASTS),10)

    def test_each_family_supports_10_distinct_contrasts(self):
        gold=constructed_NEW_TRAIN_gold120_only_for_loss()
        for family in sorted({x["family"] for x in gold}):
            cases=[x for x in gold if x["family"]==family]
            self.assertEqual(len(cases),10)
            self.assertEqual(len({x["factuality"] for x in cases}),10)
            self.assertEqual(
                sum(x["actually_transferred"] for x in cases),5)
            self.assertEqual(sum(x["rejected_directed_recipient"] for x in cases),1)
            self.assertEqual(sum(x["link"]=="NONE" for x in cases),4)

    def test_overlapping_mini_areas_are_not_one_exclusive_class(self):
        rows=constructed_NEW_TRAIN_gold120_only_for_loss()
        overlap=sum(sum(x["area_multi_hot"])>=2 for x in rows)
        self.assertGreaterEqual(overlap,72)
        self.assertTrue(any(sum(x["area_multi_hot"])==3 for x in rows))
        self.assertTrue(all(len(x["area_multi_hot"])==8 for x in rows))
        self.assertTrue(all(0<=sum(x["area_multi_hot"])<=4 for x in rows))

    def test_grouped_train_diagnostic_no_family_contamination(self):
        train,diag=grouped_train_source_only()
        self.assertEqual(len(train),90)
        self.assertEqual(len(diag),30)
        self.assertTrue(set(x["family"] for x in train)
                        .isdisjoint(x["family"] for x in diag))
        self.assertEqual(len(GROUP_TRAIN),9)
        self.assertEqual(len(GROUP_DIAG),3)

    def test_raw_module_ast_has_no_gold_import_or_model_call(self):
        p=Path("research/nra3_s20/s74_newtrain120_source_ONLY.py")
        code=p.read_text(encoding="utf8")
        tree=ast.parse(code)
        imports=[]
        for z in tree.body:
            if isinstance(z,ast.Import):
                imports.extend(item.name for item in z.names)
            elif isinstance(z,ast.ImportFrom):
                imports.append(z.module)
        self.assertTrue(all("gold" not in z.lower()
                            and "s66" not in z.lower()
                            and "eval" not in z.lower()
                            and "torch" not in z.lower()
                            for z in imports))
        self.assertNotIn("source_gold",code)
        self.assertNotIn("old43m",code)
        self.assertTrue(all(set(row)=={"row_id","family","text"}
                            for row in new_source_only_train120()))

    def test_first_read_guard_false_and_source_sha_deterministic(self):
        m=new_train120_source_gold_alignment_manifest()
        self.assertEqual(len(source_sha256()),64)
        self.assertEqual(m["source_sha256"],source_sha256())
        self.assertEqual(m["n_source_rows"],120)
        self.assertTrue(m["labels_not_independently_human_reviewed"])
        self.assertTrue(m["h8_tokenizer_source_rows_contain_only_id_family_text"])
        self.assertFalse(m["trained_new_microareas_NOT_EXECUTED"] is False)
        self.assertTrue(m["prospective_first_read_EVAL_NOT_CREATED"])
        self.assertFalse(m["candidate_accepted"])
        self.assertEqual(m["output_gate"],0)

if __name__=="__main__":
    unittest.main()
