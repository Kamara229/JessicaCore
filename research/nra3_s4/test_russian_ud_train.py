"""Offline Russian TRAIN-only source/alignment/loss and native CB59 tests."""
import dataclasses
import pathlib
import subprocess
import tempfile
import unittest

import torch

from research.nra3_s0.neural_area import NRA3Config,NRA3NeuralArea
from research.nra3_s1.physical_cb59 import encode_area,reconstruct_state
from research.nra3_s2.native_cpu import NRA3NativePackedCPU
from research.nra3_s4.russian_ud_train_only import (
    TRAIN_SPLIT,build_train,commitment,validate_train,
)
from research.nra3_s4.complete_prefix_training_contract import (
    align_complete_chat_prefix,complete_prefix_readout,
    train_only_multitask_loss,
)


class CharFastTokenizer:
    """Synthetic fast-tokenizer ABI; NOT real Calibri tokenization."""
    @staticmethod
    def apply_chat_template(messages,tokenize=False,add_generation_prompt=True):
        assert not tokenize and len(messages)==1
        return "<|user|>"+messages[0]["content"]+"<|assistant|>"

    @staticmethod
    def encode(full,add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(x)+1 for x in full]

    def __call__(self,full,add_special_tokens=False,return_offsets_mapping=False):
        return {
            "input_ids":self.encode(full,add_special_tokens),
            "offset_mapping":[(i,i+1) for i in range(len(full))],
        }


class DuplicatingTemplateTokenizer(CharFastTokenizer):
    @staticmethod
    def apply_chat_template(messages,tokenize=False,add_generation_prompt=True):
        return "<|user|>"+messages[0]["content"]+messages[0]["content"]+"<|assistant|>"


class MergedWordsTokenizer(CharFastTokenizer):
    def __call__(self,full,add_special_tokens=False,return_offsets_mapping=False):
        ret=super().__call__(full,add_special_tokens,return_offsets_mapping)
        # A single byte-pair token spanning both first and second UD words
        # cannot safely receive two incompatible sets of grammatical labels.
        start=full.index("Анна")
        ret["offset_mapping"][start]=(start,start+len("Анна отправила"))
        # Also requires a realistic matching encode returned IDs: offset
        # mapping can differ even when IDs remain unchanged, so fail closed.
        return ret


class RussianUDS4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(404004)

    @staticmethod
    def area():
        return NRA3NeuralArea(NRA3Config(hidden=32,heads=4,
                                          grammar_blocks=1,semantic_blocks=1,
                                          relation_width=24,relation_steps=2))

    def test_real_Russian_UD_TRAIN_separates_grammar_from_semantics(self):
        rows=build_train()
        self.assertEqual(len(rows),12)
        self.assertEqual(len({r.sha256() for r in rows}),12)
        self.assertTrue(all(r.split==TRAIN_SPLIT for r in rows))
        assert rows[0].words[0].deprel=="nsubj"
        self.assertEqual(rows[0].roles[0],"AGENT")
        self.assertEqual(rows[2].words[0].deprel,"nsubj")
        self.assertEqual(rows[2].roles[0],"RECIPIENT")
        self.assertEqual(rows[10].words[0].deprel,"nsubj:pass")
        self.assertEqual(rows[10].roles[0],"PATIENT")
        self.assertEqual(rows[10].roles[3],"AGENT")
        self.assertEqual(rows[0].words[3].deprel,"obj")
        self.assertEqual(rows[1].words[0].deprel,"obj")
        report=commitment(rows)
        self.assertEqual(report["row_count"],12)
        self.assertEqual(report["independent_EVAL_count"],0)
        self.assertEqual(len(report["canonical_training_rows_sha256"]),64)
        self.assertEqual(report["QAT_training_performed"],False)
        self.assertFalse(report["candidate_accepted"])

    def test_word_order_pairs_and_counterexamples_are_not_collapsed(self):
        a,b,c=build_train()[:3]
        def role_words(row,kind):
            return tuple(w.text.casefold() for w,v in zip(row.words,row.roles)
                         if v==kind)
        self.assertEqual(role_words(a,"PATIENT"),("письмо",))
        self.assertEqual(role_words(b,"PATIENT"),("письмо",))
        self.assertEqual(role_words(a,"AGENT"),("анна",))
        self.assertEqual(role_words(b,"AGENT"),("анна",))
        self.assertEqual(role_words(c,"RECIPIENT"),("павел",))
        self.assertEqual(role_words(c,"SOURCE"),("анны",))
        self.assertNotIn("AGENT",c.roles)

    def test_fast_tokenizer_wordpieces_and_complete_rightward_head(self):
        row=build_train()[0]
        tok=CharFastTokenizer()
        aligned=align_complete_chat_prefix(tok,row)
        self.assertEqual(aligned.original_prompt_sha256,row.sha256())
        self.assertEqual(len(aligned.word_alignment),len(row.words))
        self.assertEqual(aligned.complete_prefix_position,
                         aligned.tokenizer_token_count-1)
        self.assertGreater(len(aligned.word_alignment[0].token_positions),1)
        # Russian subject appears before its head predicate at word 2.
        self.assertEqual(row.words[0].head,2)
        self.assertLess(aligned.word_alignment[0].last_token,
                        aligned.word_alignment[1].last_token)
        with self.assertRaises(ValueError):
            align_complete_chat_prefix(tok,row,expected_input_ids=[1]*5)

    def test_fail_closed_duplicate_template_and_cross_word_merge(self):
        row=build_train()[0]
        with self.assertRaises(ValueError):
            align_complete_chat_prefix(DuplicatingTemplateTokenizer(),row)
        with self.assertRaises(ValueError):
            align_complete_chat_prefix(MergedWordsTokenizer(),row)
        wrong=dataclasses.replace(row,split="EVAL")
        with self.assertRaises(ValueError):
            align_complete_chat_prefix(CharFastTokenizer(),wrong)
        with self.assertRaises(ValueError):
            validate_train((row,)*12)

    def test_completed_prefix_training_objective_uses_no_future_or_extra_head(self):
        tok=CharFastTokenizer()
        row=build_train()[1]
        aligned=align_complete_chat_prefix(tok,row)
        model=self.area().train()
        x=torch.randn(1,aligned.tokenizer_token_count,320)
        output=model(x)
        decoded=complete_prefix_readout(model,output,aligned)
        words=len(row.words)
        self.assertEqual(tuple(decoded.dep_head_logits.shape),(words,words+1))
        self.assertEqual(tuple(decoded.upos_logits.shape),(words,17))
        # Later word head is legal when the *entire current prefix* exists.
        self.assertTrue(torch.isfinite(decoded.dep_head_logits[0,2]))
        losses=train_only_multitask_loss(model,x,row,aligned)
        self.assertTrue(torch.isfinite(losses.total))
        losses.total.backward()
        self.assertIsNotNone(model.grammar[0].attn.in_proj_weight.grad)
        self.assertIsNotNone(model.semantic[0].attn.in_proj_weight.grad)
        self.assertFalse(output.readout.grad_fn is None)
        self.assertEqual(model.readout.weight.numel(),24*320)
        with self.assertRaises(ValueError):
            train_only_multitask_loss(model,torch.randn(1,4,320),row,aligned)
        with self.assertRaises(ValueError):
            train_only_multitask_loss(model,x,build_train()[0],aligned)

    def test_same_complete_prefix_heads_run_in_physical_CB59_native_CPU(self):
        tok=CharFastTokenizer()
        row=build_train()[10]
        aligned=align_complete_chat_prefix(tok,row)
        original=self.area().eval()
        physical,_,_=encode_area(original)
        dense=self.area().eval()
        dense.load_state_dict(reconstruct_state(physical,original),strict=True)
        with tempfile.TemporaryDirectory() as td:
            path=pathlib.Path(td)/"native_cb59_s4.so"
            subprocess.run(["cc","-O3","-std=c11","-shared","-fPIC",
                            "research/calibri/direct_kernel_v59.c",
                            "-o",str(path)],check=True)
            native=NRA3NativePackedCPU(physical,dense.config,path)
            x=torch.randn(1,aligned.tokenizer_token_count,320)
            with torch.inference_mode():
                reference=complete_prefix_readout(dense,dense(x),aligned)
                candidate=complete_prefix_readout(native,native(x),aligned)
            for field in ("upos_logits","morph_logits","dep_head_logits",
                          "semantic_role_logits","relation_summary"):
                old=getattr(reference,field)
                new=getattr(candidate,field)
                self.assertEqual(old.shape,new.shape)
                self.assertTrue(torch.allclose(old,new,atol=4e-4,rtol=4e-4),
                                field)
            self.assertFalse(native.native_cpu_weights_dense_resident)
            self.assertEqual(list(native.parameters()),[])
            self.assertEqual(native.output_gate,0)
            self.assertFalse(native.candidate_accepted)


if __name__=="__main__":
    unittest.main()
