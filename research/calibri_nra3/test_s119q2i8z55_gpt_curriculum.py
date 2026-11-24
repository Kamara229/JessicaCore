import unittest
from pathlib import Path
from collections import Counter
from research.calibri_nra3.s119q2i8z55_gpt_curriculum import curriculum,digest,KINDS
class TestZ55GPT(unittest.TestCase):
    def test_new_and_replay(self):
        rows=curriculum()
        self.assertEqual(len(rows),42)
        self.assertEqual(sum(r["id"].startswith("S119Q2I8Z55_GPT_") for r in rows),28)
        self.assertEqual(Counter(r["kind"] for r in rows),{k:6 for k in KINDS})
        self.assertTrue(all(r["split"]=="TRAIN" for r in rows))
        self.assertEqual(digest(rows),digest(curriculum()))
    def test_no_fp32_teacher(self):
        s=Path(__file__).with_name("s119q2i8z55_physical_gpt_fullsequence.py").read_text()
        self.assertIn("teacher_135m=None,teacher_KL_weight=0.0",s)
        self.assertNotIn("AutoModelForCausalLM.from_pretrained",s)
        self.assertNotIn("source_teacher_step(",s)
        self.assertNotIn('fresh_new_rows("DEV")',s)
        self.assertIn("copy.deepcopy(child_a)",s)
if __name__=="__main__":unittest.main()
