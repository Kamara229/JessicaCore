"""S119q2c1 rigorous small synthetic counterfactual controls, no GOLD/QA."""
from __future__ import annotations

import copy
import math
import unittest
from collections import Counter

import torch

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q2a_train_source_manifest import (
    SPENT_S119N_PROMPTS, build_rows,
)
from research.calibri_nra3.s119q2c1_author_counterfactual_pairs import (
    KINDS,PAIR_BANK,rows,split,validate,canonical_bytes,sha,
    EXPECTED_SOURCE_SHA256,
)
from research.calibri_nra3.s119q2c1_real43m_onepass_counterfactual_train import (
    CONTRAST_WEIGHT,EPOCHS,LR,VOCAB_SIZE,
    paired_instant_metrics,summarize_probe,prepared_source,
)


class S119Q2C1CounterfactualControls(unittest.TestCase):
    def setUp(self):
        self.records=rows()

    def test_01_original_authored_pairs_24train_12probe_six_domains(self):
        train,probe,info=split(self.records)
        self.assertEqual(len(train),24)
        self.assertEqual(len(probe),12)
        self.assertEqual(info["TRAIN_pairs_n"],12)
        self.assertEqual(info["SOURCE_PROBE_pairs_n"],6)
        self.assertEqual(set(KINDS),set(PAIR_BANK))
        self.assertEqual(Counter(x["kind"] for x in train),
                         {k:4 for k in KINDS})
        self.assertEqual(Counter(x["kind"] for x in probe),
                         {k:2 for k in KINDS})
        self.assertTrue(all(x["pair_index"]<2 for x in train))
        self.assertTrue(all(x["pair_index"]==2 for x in probe))
        self.assertTrue(info["not_independent_unseen_generalization"])

    def test_02_sha_canonical_immutable_author_infrastructure(self):
        raw=b"".join(canonical_bytes(x) for x in self.records)
        digest=sha(raw)
        self.assertEqual(digest,validate(self.records)["source_sha256"])
        self.assertEqual(len(digest),64)
        self.assertEqual(digest,EXPECTED_SOURCE_SHA256)
        self.assertEqual(raw,b"".join(canonical_bytes(x) for x in rows()))

    def test_03_causal_flip_always_changes_explicit_answer(self):
        train,probe,_=split(self.records)
        for kind in KINDS:
            for group in range(3):
                pair=[x for x in self.records
                      if x["kind"]==kind and x["pair_index"]==group]
                self.assertEqual(len(pair),2)
                self.assertEqual({x["side"] for x in pair},{0,1})
                self.assertNotEqual(pair[0]["completion"],pair[1]["completion"])
                self.assertNotEqual(pair[0]["prompt"],pair[1]["prompt"])
                self.assertEqual(pair[0]["use"],pair[1]["use"])

    def test_04_pairwise_negation_and_arithmetic_rule_truth(self):
        neg=[x for x in self.records if x["kind"]=="negation"]
        for item in neg:
            self.assertEqual(item["completion"],
                             "Нет." if " не " in item["prompt"] else "Да.")
        calc=[x for x in self.records if x["kind"]=="arithmetic"]
        self.assertEqual([x["completion"] for x in calc],
                         ["11.","7.","20.","4.","29.","17."])

    def test_05_word_order_role_and_event_time_contrasts(self):
        role=[x for x in self.records if x["kind"]=="roles_word_order"]
        self.assertEqual([x["completion"] for x in role],
                         ["Артём.","Лена.","Максим.","Оля.","Руслан.","Вера."])
        event=[x for x in self.records if x["kind"]=="event_time"]
        self.assertTrue(all("Сначала " in x["prompt"]
                            and " затем " in x["prompt"] for x in event))
        self.assertEqual(event[0]["completion"],"Погас свет.")
        self.assertEqual(event[1]["completion"],"Открылась дверь.")

    def test_06_no_spent_q2a_or_s119n_prompt_reused(self):
        old={x["prompt"] for x in build_rows()}|set(SPENT_S119N_PROMPTS)
        self.assertFalse(old&{x["prompt"] for x in self.records})
        self.assertEqual(len(set(x["prompt"] for x in self.records)),36)

    def test_07_source_only_probe_never_gradient_train(self):
        train,probe,_=split(self.records)
        self.assertFalse({x["id"] for x in train}&{x["id"] for x in probe})
        self.assertTrue(all(x["use"]=="SOURCE_PROBE" for x in probe))
        self.assertTrue(all(x["use"]=="TRAIN" for x in train))

    def test_08_failclosed_on_one_changed_gold_or_label(self):
        changed=copy.deepcopy(self.records)
        changed[-1]["completion"]="Неверный ответ."
        with self.assertRaisesRegex(ValueError,"no changed original source"):
            validate(changed)

    def test_09_failclosed_on_removed_or_added_source_row(self):
        for changed in (self.records[:-1],self.records+[self.records[0]]):
            with self.assertRaisesRegex(ValueError,"exactly 24 TRAIN"):
                validate(changed)

    def test_10_failclosed_on_split_or_source_kind_mutation(self):
        changed=copy.deepcopy(self.records)
        changed[0]["use"]="SOURCE_PROBE"
        with self.assertRaisesRegex(ValueError,"no changed original source"):
            validate(changed)
        changed=copy.deepcopy(self.records)
        changed[0]["rights_scope"]="external_unverified"
        with self.assertRaisesRegex(ValueError,"no changed original source"):
            validate(changed)

    def test_11_exact_native_first_BPE_pair_log_odds_measurable(self):
        logits=torch.zeros(1,VOCAB_SIZE)
        logits[0,11]=5.
        logits[0,17]=1.
        good=paired_instant_metrics(logits,11,17)
        bad=paired_instant_metrics(logits,17,11)
        self.assertTrue(good["correct_ranked_over_opposing"])
        self.assertFalse(bad["correct_ranked_over_opposing"])
        self.assertTrue(good["native_distinguishing_BPE_greedy_correct"])
        self.assertFalse(bad["native_distinguishing_BPE_greedy_correct"])
        self.assertAlmostEqual(
            good["native_distinguishing_truth_minus_counterfactual_logit"],4.)
        self.assertAlmostEqual(
            bad["native_distinguishing_truth_minus_counterfactual_logit"],-4.)
        self.assertLess(good["native_distinguishing_BPE_CE"],bad["native_distinguishing_BPE_CE"])

    def test_12_reject_same_wrong_token_or_nan_full_vocab(self):
        logits=torch.zeros(1,VOCAB_SIZE)
        with self.assertRaisesRegex(GateFailure,"counterfactual"):
            paired_instant_metrics(logits,7,7)
        logits[0,0]=float("nan")
        with self.assertRaisesRegex(GateFailure,"counterfactual"):
            paired_instant_metrics(logits,7,8)

    def test_13_source_only_pair_accuracy_not_broad_intelligence(self):
        fake=[]
        for kind in KINDS:
            for side in (0,1):
                fake.append({
                    "kind":kind,"pair_index":2,"side":side,
                    "use":"SOURCE_PROBE",
                    "stats":{
                        "native_distinguishing_BPE_CE":3. if side==0 else 5.,
                        "native_distinguishing_truth_minus_counterfactual_logit":
                            1. if side==0 else -1.,
                        "correct_ranked_over_opposing":side==0,
                        "native_distinguishing_BPE_greedy_correct":False,
                    }
                })
        s=summarize_probe(fake)
        self.assertEqual(s["count_source_only_probe_prompts"],12)
        self.assertEqual(s["correct_first_token_ranks_over_opposing"],6)
        self.assertEqual(s["source_counterfactual_pairs_both_sides_correct"],0)
        self.assertAlmostEqual(s["mean_distinguishing_native_BPE_CE"],4.)
        self.assertTrue(s["same_author_source_ONLY_not_independent_quality"])

    def test_14_source_probe_cannot_be_miscounted_as_train(self):
        fake=[{
            "kind":kind,"pair_index":0,"side":side,
            "use":"TRAIN","stats":{}
        } for kind in KINDS for side in (0,1)]
        with self.assertRaisesRegex(GateFailure,"never-trained"):
            summarize_probe(fake)

    def test_15_equal_param_predeclared_train_budget(self):
        self.assertEqual(EPOCHS,1)
        self.assertEqual(LR,0.001)
        self.assertEqual(CONTRAST_WEIGHT,0.25)
        self.assertEqual(VOCAB_SIZE,49152)
        self.assertEqual(len(split(self.records)[0])*EPOCHS,24)

    def test_16_first_shared_BPE_prefix_teacher_forced_without_future_target(self):
        # Synthetic tokenizer deliberately prepends a shared native marker
        # to all answer strings to TEST the first-token COLLISION logic.
        class ToyNativeTokenizer:
            vocab_size=49152
            def apply_chat_template(self,messages,tokenize=False,
                                    add_generation_prompt=True):
                return "<user>"+messages[0]["content"]+"</user><assistant>"
            def encode(self,txt,add_special_tokens=False):
                return [1]+[ord(c) for c in txt]
            def decode(self,ids,skip_special_tokens=False):
                return "".join(chr(i) for i in ids if i != 1)
        tok=ToyNativeTokenizer()
        probe=split(self.records)[1]
        transformed=prepared_source(tok,probe)
        self.assertEqual(len(transformed),12)
        self.assertGreater(max(x["teacher_forced_shared_answer_prefix_tokens"]
                               for x in transformed),0)
        for x in transformed:
            self.assertEqual(
                len(x["prefix"]),
                x["original_question_prefix_length"]+
                x["teacher_forced_shared_answer_prefix_tokens"])
            self.assertNotEqual(
                x["native_target"],x["opposing_native_distinct_token"])
            self.assertEqual(
                x["prefix"][-x["teacher_forced_shared_answer_prefix_tokens"]:]
                if x["teacher_forced_shared_answer_prefix_tokens"] else [],
                x["teacher_forced_shared_answer_native_ids"])
        by={(x["kind"],x["pair_index"],x["side"]):x for x in transformed}
        for x in transformed:
            other=by[(x["kind"],x["pair_index"],1-x["side"])]
            self.assertEqual(x["teacher_forced_shared_answer_native_ids"],
                             other["teacher_forced_shared_answer_native_ids"])

    def test_17_paired_softplus_is_finite_and_protects_truth_direction(self):
        logits=torch.zeros(1,VOCAB_SIZE,requires_grad=True)
        loss=torch.nn.functional.cross_entropy(
            logits,torch.tensor([9])) +CONTRAST_WEIGHT*torch.nn.functional.softplus(
                logits[0,12]-logits[0,9])
        self.assertTrue(torch.isfinite(loss).item())
        grad=torch.autograd.grad(loss,logits)[0]
        self.assertLess(float(grad[0,9]),0.)
        self.assertGreater(float(grad[0,12]),0.)


if __name__=="__main__":
    unittest.main()
