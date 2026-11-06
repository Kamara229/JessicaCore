"""Synthetic deterministic acceptance-policy regressions; no actual model QA."""
from copy import deepcopy
import unittest
from v204.guards import evaluate_candidate

IDS=(0,1,6,7,12,13,18,19)


def frozen():
    first={}
    for i in range(24):
        side={}
        for side_name,offset in (("left",1),("right",2)):
            expect=(i%4)*10+offset
            side[side_name]={
                "first_token_id":expect,"expected":expect,
                "first_token_correct":True
            }
        first[i]={"domain":str(i%4),"sides":side}
    rollout={}
    for i in IDS:
        sides={}
        for side_name,start in (("left",7),("right",2)):
            gold=[start,start+2]
            pred=gold.copy()
            if i==0 and side_name=="left":
                pred=[start,start+2,start+2]
            sides[side_name]={
                "gold_tokens_with_EOS":gold,
                "generated_tokens":pred,
                "exact_with_EOS":pred==gold,
                "correct_prefix_tokens":sum(1 for a,b in zip(pred,gold) if a==b),
                "adjacent_repetition_excess":sum(a==b for a,b in zip(pred[:-1],pred[1:])),
                "stopped_at_EOS":bool(pred and pred[-1]==gold[-1])
            }
        rollout[i]={"domain":str(i%4),"sides":sides}
    return first,rollout


def gate(first=None,seq=None):
    original_first,original_seq=frozen()
    return evaluate_candidate(
        original_first,first if first is not None else deepcopy(original_first),
        original_seq,seq if seq is not None else deepcopy(original_seq),
        IDS,full_diversity_floor=0.1,
        objective_before=1.,objective_after=1.)


class TestV204GlobalTrainGuards(unittest.TestCase):
    def test_all_frozen_stable_candidate_passes(self):
        actual=gate()
        self.assertTrue(actual["accept"])
        self.assertTrue(actual["per_step_full_all_pairs_checked"])

    def test_one_token_prefix_loss_rejected_despite_repetition_drop(self):
        before,sequence=frozen()
        item=sequence[0]["sides"]["left"]
        item["generated_tokens"]=[7,8,9]
        item["correct_prefix_tokens"]=1
        item["adjacent_repetition_excess"]=0
        actual=gate(first=before,seq=sequence)
        self.assertFalse(actual["accept"])
        failure=actual["global_eight_full_sequence_guard"]
        self.assertTrue(failure["prefix_regressed"])
        self.assertFalse(failure["repetition_regressed"])
        self.assertFalse(actual["global_twentyfour_first_token_guard"]["diversity_below_precommitted_floor"])

    def test_unobserved_outside_eight_first_token_regression_rejected(self):
        first,_=frozen()
        item=first[2]["sides"]["left"]
        item["first_token_id"]=555
        item["first_token_correct"]=False
        actual=gate(first=first)
        self.assertFalse(actual["accept"])
        self.assertTrue(actual["sampled_eight_first_token_guard"]["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"])
        self.assertFalse(actual["global_twentyfour_first_token_guard"]["ACCEPT_TRAIN_ONLY_SOURCE_CANDIDATE"])

    def test_mismatched_full_sequence_subset_is_rejected(self):
        first,sequence=frozen()
        del sequence[0]
        original_first,original_sequence=frozen()
        with self.assertRaises(AssertionError):
            evaluate_candidate(original_first,first,original_sequence,sequence,IDS,0.1,1.,1.)

    def test_local_train_objective_regression_is_rejected(self):
        before_first,before_seq=frozen()
        actual=evaluate_candidate(before_first,deepcopy(before_first),
            before_seq,deepcopy(before_seq),IDS,0.1,1.,1.02)
        self.assertFalse(actual["accept"])
        self.assertFalse(actual["TRAIN_loss_no_more_than_1percent_increase"])


if __name__=="__main__":
    unittest.main()
