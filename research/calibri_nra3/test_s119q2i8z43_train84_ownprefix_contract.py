"""S43 source/test contracts: no dev/gold, no optimizer, original 43M frozen."""
import ast
import unittest
from pathlib import Path

FILE=Path(__file__).with_name("s119q2i8z43_true43m_legal84_onpolicy_greedy_coverage.py")

def run_source():
    program=ast.parse(FILE.read_text(encoding="utf8"))
    fn=next(node for node in program.body if isinstance(node,ast.FunctionDef)
            and node.name=="run")
    return ast.get_source_segment(FILE.read_text(encoding="utf8"),fn)

class TestS43StrictTrainingOnlyOnPolicy(unittest.TestCase):
    def test_compilation_and_source_only_no_human_DEV_forwards(self):
        text=run_source()
        self.assertIn('fresh_new_rows("TRAIN")',text)
        self.assertNotIn('fresh_new_rows("DEV")',text)
        self.assertNotIn("for row in fresh_dev:",text)
        self.assertNotIn("fit_controls(",text)
        self.assertNotIn("optimizer.step(",text)
        self.assertIn('r["split"]=="TRAIN"',text) if False else None
        self.assertIn('row["split"]=="TRAIN"',text)
        self.assertIn('"no_new_DEV_or_GOLD_evaluation":True',text)
        self.assertIn('"no_fitted_shadow_training_or_optimizer_steps":True',text)
        self.assertIn('"candidate_accepted":False,"output_gate":0',text)

    def test_reachable_and_counterfactual_separated(self):
        text=run_source()
        for marker in (
            'prev_matches=own==gold[:position-1]',
            '"prior_prefix_gold":prev_matches',
            '"natural_prior_own_prefix_exactly_matches_gold_count":len(reachable)',
            '"own_prefix_gold_target_in_top64_all_generated"',
            '"natural_prior_correct_and_target_top64_count"',
            '"teacher_gold_prefix_top64_eligible_in_archived_Z41_TRAIN84"',
            'rank=int((logits>logits[target]).sum().item())+1',
            "original_forwards+=called[0]",
            'called[0]==len(own)',
        ):
            with self.subTest(marker=marker):self.assertIn(marker,text)

    def test_previous_84_source_SHA_and_original_S142_parity(self):
        text=run_source()
        for marker in (
            'len(fresh_train)==84',
            'len({r["prompt_sha256"] for r in fresh_train})==84',
            'a66833e8f4e83e4ba080dc3db0b3032a3315bcfc2fe7cee473e7e9ea7569023e',
            'archived["original43M_49k_native_FORWARDS"]==913',
            'teacher=={"2":51,"3":72,"4":65}',
            'old_original_param_digest(model)==original_digest',
            '"original_S142_43M_GT12_Core_NRA3_NPA1_and_Z10_A_B_unchanged":True',
        ):
            with self.subTest(marker=marker):self.assertIn(marker,text)

if __name__=="__main__":
    unittest.main()
