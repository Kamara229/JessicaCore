"""NRA3 S92 source-only original43M 12layer physical ABI preflight.

No original Calibri weights/teacher gold loaded by these AST tests.
Actual CPU replay + all12 saved requires separate GH Actions job.
"""
import ast
import unittest
from pathlib import Path
from research.nra3_s20.s83_NEW_DIVERSE72_source_ONLY_never_use_S79_spent import (
    source_only_new72,source_only_new72_sha256,
)

P=Path("research/calibri_nra3/s92_real43m_GT_all12_newtrain72_sourceonly.py")


class TestS92PhysicalAll12SourceContract(unittest.TestCase):
    def test_only_frozen_S83_train72_source_sha(self):
        rows=source_only_new72()
        self.assertEqual(len(rows),72)
        self.assertEqual(source_only_new72_sha256(),
            "285475efb25a1310ce87fa5ffc9e8ec99dd5f129d147b0edf4afa7195f43a62a")
        self.assertTrue(all(set(x)=={"row_id","group","text"} for x in rows))

    def test_all_model_imports_without_gold_or_spent_test(self):
        t=ast.parse(P.read_text(encoding="utf8"))
        imports=[z.module or "" for z in ast.walk(t)
                 if isinstance(z,ast.ImportFrom)]
        self.assertTrue(any("s91_neural_multilayer_event_graph" in x for x in imports))
        self.assertTrue(any("s83_NEW_DIVERSE72_source_ONLY" in x for x in imports))
        self.assertFalse(any("gold" in x.casefold() or
                             "s79_new_novel" in x.casefold() or
                             "s86_newsynthetic" in x.casefold() or
                             "s88_s85" in x.casefold()
                             for x in imports))

    def test_register_embedded_weights_after_protected_model_check_and_layer8_identity(self):
        code=P.read_text(encoding="utf8")
        self.assertIn("old_calibri_parameter_sha(model)",code)
        self.assertIn("child.attach_to_RESEARCH_calibri(model)",code)
        self.assertIn("child.read_only_all12_source_scope(model)",code)
        self.assertIn("layer8.capture(model,text,ids)",code)
        self.assertIn("torch.equal(observed[0,7],captured.hidden320_fp16)",code)
        self.assertIn("model.nra3_neural_weights.",code)
        self.assertIn("all_old_Calibri43M_weights_unchanged_sha256",code)

    def test_only_save_after_616_baseline_has_been_verified(self):
        t=ast.parse(P.read_text(encoding="utf8"))
        funcs={n.name for n in t.body if isinstance(n,ast.FunctionDef)}
        self.assertIn("real_all12_source_capture_AFTER_original_616",funcs)
        self.assertIn("save_only_AFTER_original616_digest_verified",funcs)
        self.assertIn("old_calibri_parameter_sha",funcs)
        s=P.read_text(encoding="utf8")
        self.assertIn("new_integrated_616_J7_not_run",s)
        self.assertIn("all_in_bpp_le07_and_CPU_ge90_NOT_certified",s)
        self.assertIn("S83_CONSTRUCTED_GOLD_not_imported_in_original_source_H8",s)
        self.assertNotIn("AdamW",s)


if __name__=="__main__":
    unittest.main()
