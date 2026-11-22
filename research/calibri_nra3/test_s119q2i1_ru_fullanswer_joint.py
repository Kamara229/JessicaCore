"""S119q2i1 native 43M full-answer EOS TRAIN safety and causality contracts."""
from __future__ import annotations

import copy
from collections import Counter
from unittest import TestCase,main
import torch
from torch import nn

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q2i1_authored_ru_train_source import (
    CASES,KINDS,SOURCE_LICENSE,TRAIN_PER_KIND,DEV_PER_KIND,
    SEALED_PER_KIND,rows,provenance,hash_rows,
    protected_eval_digest,
)
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import (
    NativeHeadAlignment,temporary_learned_native_head,
    verified_native_answer,HEAD_PARAMETER_COUNT,N_TRAIN,N_DEV,EPOCHS,
    MAX_ANSWER_NATIVE,DEV_MAX_GEN_NATIVE,
    HEAD_LEARNING_RATE,CHILD_LEARNING_RATE,STAGE,
)


class ToyNative:
    vocab_size=49152
    eos_token_id=2
    def apply_chat_template(self,messages,tokenize=False,
                            add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return "<u>"+messages[0]["content"]+"</u><a>"
    def encode(self,txt,add_special_tokens=False):
        assert not add_special_tokens
        return [ord(c) for c in txt]
    def decode(self,ids,skip_special_tokens=False):
        return "".join(chr(x) for x in ids if x!=2)


class ToyFrozenOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head=nn.Linear(320,49152,bias=False)
        self.lm_head.requires_grad_(False)
    def forward(self,x):
        return self.lm_head(x)


class S119Q2I1NativeLanguageTrainingContracts(TestCase):
    def test_01_truly_new_original_author_train_and_dev_disjoint(self):
        meta=provenance()
        self.assertEqual(meta["origin"],SOURCE_LICENSE)
        self.assertEqual(meta["TRAIN_rows"],36)
        self.assertEqual(meta["DEV_rows"],12)
        self.assertTrue(meta["old_sources_and_spent_q2g_disjoint"])
        self.assertTrue(meta["original_author_training_examples_only"])
        self.assertTrue(meta["no_external_independent_blind_QA_claim"])

    def test_02_six_kinds_balanced_source_counts(self):
        tr=rows("TRAIN"); dv=rows("DEV")
        self.assertEqual(Counter(x["kind"] for x in tr),
                         {k:TRAIN_PER_KIND for k in KINDS})
        self.assertEqual(Counter(x["kind"] for x in dv),
                         {k:DEV_PER_KIND for k in KINDS})
        self.assertFalse({x["prompt"] for x in tr}&
                         {x["prompt"] for x in dv})
        self.assertEqual(len({x["prompt"] for x in tr+dv}),48)

    def test_03_provenance_train_hash_frozen(self):
        meta=provenance()
        self.assertEqual(meta["TRAIN_sha256"],hash_rows(rows("TRAIN")))
        self.assertEqual(meta["DEV_sha256"],hash_rows(rows("DEV")))
        self.assertEqual(len(meta["TRAIN_sha256"]),64)
        self.assertEqual(len(meta["DEV_sha256"]),64)

    def test_04_sealed_eval_never_returned_to_training_API(self):
        self.assertEqual(SEALED_PER_KIND,2)
        self.assertEqual(provenance()["SEALED_EVAL_count_NEVER_OPENED"],12)
        with self.assertRaisesRegex(ValueError,"SEALED_EVAL"):
            rows("SEALED_EVAL")
        with self.assertRaisesRegex(ValueError,"SEALED_EVAL"):
            rows("EVAL")
        self.assertEqual(len(protected_eval_digest()),64)

    def test_05_original_author_corpus_complete(self):
        self.assertEqual(len(KINDS),6)
        self.assertTrue(all(len(CASES[k])==10 for k in KINDS))
        self.assertTrue(all(len(prompt)>14 and len(answer)>1
                            for k in KINDS for prompt,answer in CASES[k]))

    def test_06_original_native_head_parameter_count_budget(self):
        h=NativeHeadAlignment()
        self.assertEqual(HEAD_PARAMETER_COUNT,247040)
        self.assertEqual(sum(p.numel() for p in h.parameters()),247040)
        self.assertEqual(h.output_gate,0)
        self.assertFalse(h.candidate_accepted)
        self.assertFalse(h.activation_allowed)

    def test_07_native_head_start_at_exact_zero_delta(self):
        h=NativeHeadAlignment()
        output=h(torch.randn(1,3,320))
        self.assertEqual(tuple(output.shape),(1,3,49152))
        self.assertTrue(torch.equal(output,torch.zeros_like(output)))

    def test_08_head_gradients_and_unfrozen_new_neural_parameters(self):
        h=NativeHeadAlignment()
        z=h(torch.randn(1,2,320))
        objective=z[0,-1,75]
        objective.backward()
        self.assertGreater(float(h.up.weight.grad.abs().sum()),0)
        self.assertGreater(float(h.output_bias.grad.abs().sum()),0)

    def test_09_real_original_lm_head_hook_adds_learned_delta_only(self):
        model=ToyFrozenOriginal()
        h=NativeHeadAlignment()
        x=torch.randn(1,3,320)
        initial=model(x).detach().clone()
        with torch.no_grad():
            h.output_bias[100]=1.5
        with temporary_learned_native_head(model,h) as proof:
            got=model(x)
        self.assertEqual(proof["original_head_calls"],1)
        self.assertAlmostEqual(float((got-initial)[0,-1,100]),1.5,places=5)
        self.assertTrue(torch.equal(model(x).detach(),initial))
        self.assertFalse(model.lm_head.weight.requires_grad)

    def test_10_hook_restored_even_after_runtime_exception(self):
        model=ToyFrozenOriginal()
        h=NativeHeadAlignment()
        x=torch.randn(1,2,320)
        original=model(x).detach().clone()
        with self.assertRaisesRegex(RuntimeError,"forced"):
            with temporary_learned_native_head(model,h):
                model(x)
                raise RuntimeError("forced")
        self.assertTrue(torch.equal(model(x).detach(),original))
        self.assertEqual(len(model.lm_head._forward_hooks),0)

    def test_11_cannot_enable_modified_head_in_production(self):
        h=NativeHeadAlignment()
        h.candidate_accepted=True
        with self.assertRaisesRegex(GateFailure,"original frozen"):
            h(torch.randn(1,2,320))

    def test_12_original_full_native_no_first_only_training(self):
        self.assertEqual(N_TRAIN,36)
        self.assertEqual(N_DEV,12)
        self.assertEqual(EPOCHS,2)
        self.assertEqual(HEAD_LEARNING_RATE,0.018)
        self.assertEqual(CHILD_LEARNING_RATE,0.0003)
        self.assertEqual(DEV_MAX_GEN_NATIVE,16)
        self.assertEqual(MAX_ANSWER_NATIVE,48)

    def test_13_teacher_forced_gold_answer_has_explicit_REAL_eos(self):
        tok=ToyNative()
        row=rows("TRAIN")[0]
        from unittest.mock import patch
        with patch(
            "research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training.strict_chat_source_ids",
            return_value=tok.encode(tok.apply_chat_template(
                [{"role":"user","content":row["prompt"]}]))
        ),patch(
            "research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training.prompt_features",
            return_value=torch.zeros(512)
        ):
            packed=verified_native_answer(tok,row)
        self.assertEqual(packed["split"],"TRAIN")
        self.assertEqual(packed["answer_native_with_eos"][-1],2)
        self.assertNotIn(2,packed["answer_native_without_eos"])
        self.assertEqual(
            packed["answer_native_with_eos"][:-1],
            packed["answer_native_without_eos"])
        self.assertFalse(
            packed["prefix"]==packed["prefix"]+
            packed["answer_native_with_eos"])

    def test_14_never_prepare_sealed_gold_or_old_probes(self):
        tok=ToyNative()
        row=copy.deepcopy(rows("TRAIN")[0])
        row["split"]="SEALED_EVAL"
        with self.assertRaisesRegex(GateFailure,"NEVER SEALED_EVAL"):
            verified_native_answer(tok,row)
        row=copy.deepcopy(rows("DEV")[0])
        row["origin"]="unknown_licensed_content"
        with self.assertRaisesRegex(GateFailure,"NEVER SEALED_EVAL"):
            verified_native_answer(tok,row)

    def test_15_fail_closed_real_head_requires_frozen_original_lm_head(self):
        h=NativeHeadAlignment()
        fake=nn.Linear(320,49152)
        with self.assertRaisesRegex(GateFailure,"ORIGINAL native lm_head"):
            with temporary_learned_native_head(fake,h):
                pass

    def test_16_implemented_model_is_training_only_not_quality_cert(self):
        self.assertIn("FULL_NATIVE_BPE_EOS_TRAIN",STAGE)
        self.assertTrue(all(x["split"]=="TRAIN" for x in rows("TRAIN")))
        self.assertTrue(all(x["split"]=="DEV" for x in rows("DEV")))


if __name__=="__main__":
    main()
