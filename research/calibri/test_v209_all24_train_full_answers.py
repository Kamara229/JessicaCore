"""v209 synthetic read-only full-answer TRAIN correctness regression checks."""
import copy
import unittest
from v209.train_only_full_answers import compare


def probe():
    full={}
    first={}
    for i in range(24):
        side={}
        tokens={}
        for name,expected in (("left",1),("right",2)):
            gold=[expected,4]
            side[name]={"gold_tokens_with_EOS":gold,"generated_tokens":list(gold),
                "exact_with_EOS":True,"correct_prefix_tokens":2,
                "adjacent_repetition_excess":0,"stopped_at_EOS":True}
            tokens[name]={"first_token_id":expected,"expected":expected,
                          "first_token_correct":True}
        full[i]={"domain":f"domain_{i//6}","sides":side}
        first[i]={"domain":f"domain_{i//6}","sides":tokens}
    return full,first


class CheckReadOnlyTrainReports(unittest.TestCase):
    def test_all_48_complete_TRAIN_answers_and_transition_grades(self):
        original,fo=probe()
        changed,f1=copy.deepcopy(original),copy.deepcopy(fo)
        changed[0]["sides"]["left"].update(
            generated_tokens=[3,3],exact_with_EOS=False,
            correct_prefix_tokens=0,adjacent_repetition_excess=1,
            stopped_at_EOS=False)
        f1[0]["sides"]["left"].update(first_token_id=3,first_token_correct=False)
        result=compare(original,changed,fo,f1)
        self.assertEqual(result["frozen_FULL_TRAIN_answer_count"],48)
        self.assertEqual(result["v186_full_TRAIN_exact"],48)
        self.assertEqual(result["v206_full_TRAIN_exact"],47)
        self.assertEqual(result["net_exact_answers_gained"],-1)
        self.assertEqual(result["v186_full_TRAIN_correct_prefix_tokens"],96)
        self.assertEqual(result["v206_full_TRAIN_correct_prefix_tokens"],94)
        self.assertEqual(result["v206_adjacent_generated_repetitions"],1)
        self.assertEqual(result["full_answer_exact_transitions"],
                         {"exact -> exact":47,"exact -> nonexact":1})
        self.assertTrue(result["NO_model_fitting_no_checkpoint_selection"])
        self.assertTrue(result["NOT_a_certification_of_FP32_parity"])

    def test_hard_fail_if_a_frozen_TRAIN_gold_changes(self):
        original,fo=probe()
        changed=copy.deepcopy(original)
        changed[0]["sides"]["left"]["gold_tokens_with_EOS"]=[3,4]
        with self.assertRaises(AssertionError):
            compare(original,changed,fo,fo)

    def test_hard_fail_if_TRAIN_pair_missing(self):
        original,fo=probe()
        changed=copy.deepcopy(original)
        del changed[23]
        with self.assertRaises(AssertionError):
            compare(original,changed,fo,fo)

    def test_hard_fail_if_first_token_expected_drift(self):
        original,fo=probe()
        changed=copy.deepcopy(fo)
        changed[5]["sides"]["left"]["expected"]=3
        with self.assertRaises(AssertionError):
            compare(original,original,fo,changed)


if __name__=="__main__":
    unittest.main()
