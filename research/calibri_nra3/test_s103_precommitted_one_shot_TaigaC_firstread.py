"""S103 one-shot prospectively sealed Taiga train-c original-source-only protocol."""
import ast
import unittest
from pathlib import Path
from research.calibri_nra3 import s103_precommit_ONE_SHOT_TaigaC_source_gold_WALL as s103

P=Path("research/calibri_nra3/s103_precommit_ONE_SHOT_TaigaC_source_gold_WALL.py")


class TestS103ProspectiveFirstReadPrecommit(unittest.TestCase):
    def test_frozen_original_source_partition_never_seen_by_S96_S99_S102(self):
        self.assertEqual(s103.STAGE,"CALIBRI_NRA3_S103_ONESHOT_TAIGA_TRAINC_FIRSTREAD_SEALED")
        self.assertEqual(s103.TRAIN_FILE,"ru_taiga-ud-train-c.conllu")
        self.assertEqual(s103.TRAIN_GIT_BLOB,
            "16ddc9c25dc1ab56d809c4b155f4eb18942b0e8f")
        self.assertEqual(s103.SOURCE_FILE,
            "NRA3_S103_TAIGA_TRAINC256_FIRSTREAD_SOURCE_ONLY_NO_GOLD.json")
        self.assertEqual(s103.GOLD_FILE,
            "NRA3_S103_TAIGA_TRAINC256_FIRSTREAD_UD_ALLVERB_GOLD_SEPARATE.json")

    def test_frozen_weight_SHAs_are_precommitted_before_firstread(self):
        s=P.read_text()
        self.assertIn("c77a6964ae4e02cb3aac98d69978a25cec2b491a9624cc9668b3b307699bff8a",s)
        self.assertIn("f40dddbc03282a6c45c4f807746275d76a4ba776aa613247e73fbdf7f43329df",s)
        self.assertIn("no_model_selection_or_hparam_tuning_from_one_shot_firstread",s)
        self.assertIn("train_c_gold_must_NEVER_enter_training",s)

    def test_no_source_EVAL_or_train_c_runtime_GOLD_imports(self):
        tree=ast.parse(P.read_text())
        modules=[x.module or "" for x in ast.walk(tree)
                 if isinstance(x,ast.ImportFrom)]
        self.assertFalse(any("s79_" in x.lower() or "s86_" in x.lower()
                             or "s96_train" in x.lower()
                             or "s102_train" in x.lower()
                             for x in modules))
        self.assertFalse(any(x.__class__.__name__ in ("Call",)
            and isinstance(x.func,ast.Name) and x.func.id in ("torch","model")
            for x in ast.walk(tree)))
        self.assertEqual(s103.LICENSE,"CC BY-SA 4.0")

    def test_distinct_source_and_teacher_gold_file_research_only(self):
        s=P.read_text()
        self.assertIn("SOURCE_FILE",s)
        self.assertIn("GOLD_FILE",s)
        self.assertIn("source_ONLY_does_not_contain_any_UD_labels",s)
        self.assertIn("output_gate",s)
        self.assertIn("activation_allowed",s)


if __name__=="__main__":
    unittest.main()
