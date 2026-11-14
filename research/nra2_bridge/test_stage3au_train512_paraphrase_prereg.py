"""S3AU split, text-order role label and no DEV/EVAL leakage checks."""
import unittest
from research.nra2_bridge.stage3au_train512_paraphrase_prereg import (
    build_rows, commitment, validate, STYLES, DESTINATION_FIRST,
)
from research.nra2_bridge.stage3aq_train_dev_futureeval_prereg import (
    build_rows as source_rows, TRAIN, DEV, EVAL,
)


class Stage3AUGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=build_rows()
        cls.parent=source_rows()
        cls.prereg=commitment(cls.rows)

    def test_training_only_512_distinct(self):
        self.assertTrue(validate(self.rows))
        self.assertEqual(len(self.rows),512)
        self.assertEqual(len(set(r.graph_id for r in self.rows)),64)
        self.assertEqual(len(set(r.prompt_sha256 for r in self.rows)),512)

    def test_no_dev_or_future_eval_prompts_or_graph_topology(self):
        held=[r for r in self.parent if r.split in (DEV,EVAL)]
        source=[r for r in self.parent if r.split==TRAIN]
        self.assertEqual(len(held),256)
        self.assertFalse({r.prompt_sha256 for r in self.rows}&
                         {r.prompt_sha256 for r in held})
        self.assertFalse({r.topology_key for r in self.rows}&
                         {r.topology_key for r in held})
        self.assertEqual({r.topology_key for r in self.rows},
                         {r.topology_key for r in source})

    def test_role_order_yes_no_balance_all_8_styles(self):
        self.assertEqual(len(STYLES),8)
        self.assertEqual(DESTINATION_FIRST,(1,3,5,7))
        for i in range(8):
            r=[r for r in self.rows if r.query_style==i]
            self.assertEqual(len(r),64)
            self.assertEqual(sum(a.reachable for a in r),32)

    def test_immutable_prereg_and_no_model(self):
        r=self.prereg
        self.assertEqual(r["TRAIN512_count"],512)
        self.assertEqual(len(r["commitment_sha256"]),64)
        self.assertEqual(r["S3AQ_parent_preregistered_SHA256"],
            "c55799793263d5b4175fdfd62c293bcc2cb1794c73fa6ba856cf8b0a961f7839")
        self.assertTrue(r["future_EVAL128_never_prefilled_scored_or_trained"])
        self.assertTrue(r["no_new_model_weights_or_training_or_scores"])
        self.assertEqual(r["output_gate"],0.0)


if __name__=="__main__":
    unittest.main()
