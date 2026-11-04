"""No QA nor unseen source read; test optimistic teacher-top1 support ceiling."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from v153.codec import Physical
from v165.protocol import (check,TOPK,ARMS,FIT_DOMAINS,TOKENS_PER_DOMAIN,
                           FULL_PARENT_ARCHIVE_BYTES)
from v165.evidence import verify_bad_real_generation
from v165.scoring import one_window,aggregate


class OriginalGreenTunnelFeasibility(unittest.TestCase):
    def test_only_frozen_train_fit_and_fixed_upper_bound_K(self):
        self.assertTrue(check())
        self.assertEqual(TOPK,(1,8,32,128,512,2048))
        self.assertEqual(ARMS,("original_native_CB59","fully_paid_layer0_CB161"))
        self.assertEqual(FIT_DOMAINS,("wiki_fit","news_fit"))
        self.assertEqual(TOKENS_PER_DOMAIN,252)
        self.assertEqual(FULL_PARENT_ARCHIVE_BYTES,11292174)

    def test_exact_gold_and_teacher_top1_recovery_is_ONLY_an_upper_bound(self):
        teacher=torch.zeros((1,64,49152),dtype=torch.float32)
        student=torch.zeros((1,64,49152),dtype=torch.float32)
        ids=torch.full((1,64),999,dtype=torch.long)
        teacher[0,:-1,999]=20.
        student[0,:-1,888]=11.
        student[0,:-1,999]=10.
        student[0,:-1,777]=9.
        window=one_window(teacher,student,ids)
        self.assertTrue(all(r==2 for r in window[
            "strict_gt_rank_for_teacher_TOP1_optimistic_ties"]))
        self.assertEqual(window["student_TOP1_matches_original_teacher_TOP1"],0)
        self.assertEqual(window["teacher_TOP1_is_true_gold_token"],63)
        self.assertEqual(window["student_TOP1_is_true_gold_token"],0)
        parsed=aggregate([window]*4)
        caps=parsed["topK_support_information_theoretic_reranker_ceiling_not_performance"]
        self.assertEqual(caps["1"]["teacher_TOP1_recovery_upper_bound_fraction"],0.)
        self.assertEqual(caps["8"]["teacher_TOP1_recovery_upper_bound_fraction"],1.)
        self.assertTrue(parsed["not_an_actual_Green_Tunnel_run"])
        # Optimistic ties: teacher token may tie for student best position.
        student[0,:-1,999]=11.
        tied=one_window(teacher,student,ids)
        self.assertTrue(all(r==1 for r in tied[
            "strict_gt_rank_for_teacher_TOP1_optimistic_ties"]))
        self.assertTrue(all(t>=2 for t in tied[
            "student_logit_ties_at_teacher_TOP1"]))
        with self.assertRaises(ValueError):
            one_window(teacher,student[:,:-1,:],ids)

    def test_source_evidence_must_include_actual_bad_four_greedy_responds(self):
        physical=Physical(b"fixed old source",0,{},FULL_PARENT_ARCHIVE_BYTES,"a"*64)
        old={
           "stage":"v163_REAL_PAID_ORIGINAL_BLOCK0_NEW_QA_AND_GREEDY_GENERATION",
           "immutable_physical_archive_SHA256":physical.sha256,
           "genuine_autoregressive_text_and_speed":{
             "genuine_full_model_autoregressive_measures":{
               "original_CB59":{
                 "distinct_token_ratio_over_all_four_responses":.0625,
                 "genuine_full_model_greedy_prompts":4,
               },
               "fully_paid_CB161_block0":{
                 "distinct_token_ratio_over_all_four_responses":.0859375,
                 "genuine_full_model_greedy_prompts":4,
               },
             }
           },
        }
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/"actual_v163_greedy.json"
            path.write_text(json.dumps(old))
            result=verify_bad_real_generation(physical,path)
            self.assertTrue(result["real_language_repetition_not_recovered"])
            old["genuine_autoregressive_text_and_speed"][
                "genuine_full_model_autoregressive_measures"][
                "fully_paid_CB161_block0"][
                "distinct_token_ratio_over_all_four_responses"]=.5
            path.write_text(json.dumps(old))
            with self.assertRaises(ValueError):
                verify_bad_real_generation(physical,path)


if __name__=="__main__":
    unittest.main()
