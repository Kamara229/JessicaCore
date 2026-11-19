"""NRA2 S75: real 12-layer original Calibri43M GT read-only NEW TRAIN120 H8.

ONLY imports S74 SOURCE-ONLY rows. Never imports constructed TRAIN labels,
S74 gold module, S66 held gold, QA/EVAL32/EVAL128, optimizer/trainer, nor
an answer class. Invoked from S66 physical CPU wrapper strictly AFTER its
original protected 616 J7 old+new generated token replay and original S190
32 actual token captures. If protected baseline ultimately fails post-hoc,
its caller must fail before output artifact upload.

Original unchanged genuine BPE token IDs + native fast-tokenizer Unicode
offsets yield S51 fractional BPE word320, S18 18 source cues and S48
frozen text-only quote sidecar -> exactly 339D. Every unsafe shared BPE
word marked (not silently treated causal). Full future text is visible,
so these H8 representations are source-only OFFLINE/TRAIN and are NOT
autoregressive streaming-causal evidence.
"""
from __future__ import annotations

from hashlib import sha256

import torch
from safetensors.torch import save_file,load_file
from pathlib import Path

from calibri2.s160 import real_parallel_slots_j7 as s160
from v176.operators import clear
from research.nra3_s5.frozen_layer8_read_port import FrozenLayer8ReadPort
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s49_real43m_gt_quote_sidecar_zero_gate import (
    Frozen43MGreenTunnelQuotePort, frozen_native_manifest_digest,
)
from research.nra3_s20.s51_fractional_native_bpe_overlap import (
    align_offsets_from_raw_text,fractional_native_word_pool,
)
from research.nra3_s20.s74_newtrain120_source_ONLY import (
    new_source_only_train120,source_sha256,GROUP_TRAIN,GROUP_DIAG,
)

EXPECTED_S74_SOURCE_SHA="5ff7f5681b0130d4c230fe5b3513038e4d6f3487a3b71964a7ad79e53f50b351"
STAGE="NRA2_S75_REAL_ORIGINAL43M_GT_PROTECTED616J7_NEW_TRAIN120_NATIVE_BPE_H8_SOURCE_ONLY"
TENSOR_NAME="NRA2_S75_GENUINE43M_GT_NEWTRAIN120_WORD339_AND_UNSAFE.safetensors"
REPORT_NAME="NRA2_S75_GENUINE43M_GT_NEWTRAIN120_SOURCE_PROVENANCE.json"


def require(condition,why):
    if not bool(condition):
        raise ValueError("NRA2 S75 original43M GT source-only capture: "+why)


def source_offset_alignment(tok,text,native_ids):
    """Original chat template once, native token IDs *untouched*, S51 offsets."""
    original_chat=tok.apply_chat_template(
        [{"role":"user","content":text}],tokenize=False,
        add_generation_prompt=True)
    require(original_chat.count(text)==1,
            "source occurs exactly once in original Calibri chat")
    encoded=tok(original_chat,add_special_tokens=False,
                return_offsets_mapping=True)
    orig_ids=tuple(int(z) for z in tok.encode(
        original_chat,add_special_tokens=False))
    observed_ids=tuple(int(z) for z in encoded["input_ids"])
    offsets=tuple((int(a),int(b)) for a,b in encoded["offset_mapping"])
    require(orig_ids==observed_ids==tuple(native_ids)
            and len(offsets)==len(orig_ids),
            "genuine original Calibri BPE IDs or native offsets drift")
    alignment=align_offsets_from_raw_text(
        text,orig_ids,offsets,original_chat.index(text),len(original_chat))
    require(alignment.original_token_ids==orig_ids
            and alignment.invalid_or_uncovered_characters==0
            and not alignment.source_gold_loaded
            and not alignment.causal_token_training_allowed,
            "S51 original native BPE fractional source ABI")
    return alignment


def capture_S74_new_train120_original43m_cpu_source_only(
        model,sources,tok,bridge,tunnel,expansion):
    rows=new_source_only_train120()
    require(source_sha256()==EXPECTED_S74_SOURCE_SHA
            and len(rows)==120
            and all(set(r)=={"row_id","family","text"} for r in rows)
            and all(r["family"] in set(GROUP_TRAIN)|set(GROUP_DIAG)
                    for r in rows),
            "source-only S74 exact immutable 120 TRAIN source before gold")
    require(len(model.model.layers)==12 and not model.training
            and next(model.parameters()).device.type=="cpu"
            and not tunnel._active and not expansion._active
            and not torch.cuda.is_available(),
            "original frozen physical Calibri43M+GT CPU no pretrained changes")
    collector=FrozenLayer8ReadPort()
    alignments=[]
    for row in rows:
        text=row["text"]
        ids=list(s160.prompt_tokens(tok,text))
        m=source_offset_alignment(tok,text,ids)
        features=torch.as_tensor(
            s160.prompt_only_vector(tok,text),dtype=torch.float32)
        require(features.shape==(512,)
                and bool(torch.isfinite(features).all()),
                "original S95 source-only 512D route exact")
        with (torch.inference_mode(),
              expansion.prefill_only(model),
              s160.prefill_only_tunnel(model,tunnel)):
            with s160.frozen_cpu_phase_scope(
                    model,bridge,"S93",features,phase="prefill_only"):
                with collector.capture(model,text,ids):
                    clear(sources)
                    output=model(
                        input_ids=torch.tensor([ids],dtype=torch.long),
                        past_key_values=None,use_cache=True,
                        output_hidden_states=True)
                    require(output.past_key_values is not None
                            and output.hidden_states[-1].shape[-1]==320,
                            "original physical cached 12-layer prefill")
                    del output
                    clear(sources)
        alignments.append(m)
    texts=tuple(r["text"] for r in rows)
    require(collector.observed_calls==120 and len(alignments)==120,
            "all 120 original physically captured BPE originals")
    wordcounts=[len(m.raw_words) for m in alignments]
    max_words=max(wordcounts)
    require(max_words<=40 and min(wordcounts)>=3,
            "original S51 raw-word geometry admissible")
    raw338=torch.zeros((120,max_words,338),dtype=torch.float32)
    valid=torch.zeros((120,max_words),dtype=torch.bool)
    unsafe=torch.zeros((120,max_words),dtype=torch.bool)
    n_native_tokens=0
    n_unsafe_words=0
    n_shared_pieces=0
    for i,(row,m) in enumerate(zip(rows,alignments)):
        sample=collector.samples[sha256(row["text"].encode("utf-8")).hexdigest()]
        native=sample.hidden320_fp16
        require(tuple(sample.token_ids)==m.original_token_ids
                and native.shape==(m.original_token_count,320)
                and native.dtype==torch.float16
                and not native.requires_grad,
                "actual original Calibri43M native FP16 H8 source matching")
        before=native.clone()
        pooled=fractional_native_word_pool(native,m)
        require(torch.equal(native,before)
                and pooled.shape==(wordcounts[i],320),
                "S51 fractional BPE must be read-only and complete")
        words=raw_surface_grammar(row["text"])
        require(len(words)==wordcounts[i], "raw-only source words/offsets drift")
        raw338[i,:len(words),:320]=pooled
        raw338[i,:len(words),320:338]=torch.tensor(
            [w.features for w in words],dtype=torch.float32)
        valid[i,:len(words)]=True
        if m.ambiguous_raw_word_ids:
            unsafe[i,list(m.ambiguous_raw_word_ids)]=True
        n_native_tokens+=m.original_token_count
        n_unsafe_words+=len(m.ambiguous_raw_word_ids)
        n_shared_pieces+=len(m.shared_native_token_ids)
    require(not bool((unsafe&~valid).any())
            and bool(torch.isfinite(raw338).all())
            and all(row["family"] in set(GROUP_TRAIN)|set(GROUP_DIAG)
                    for row in rows),
            "S75 original native word mask or offset overlap invalid")
    before_sha=frozen_native_manifest_digest(raw338,valid)
    port=Frozen43MGreenTunnelQuotePort()
    same=port.forward_original_unchanged(raw338,valid,texts)
    extended,quote_sidecar=port.shadow_augmented_features_only(raw338,valid,texts)
    require(same is raw338 and same.data_ptr()==raw338.data_ptr()
            and extended.shape==(120,max_words,339)
            and torch.equal(extended[...,:338],raw338)
            and before_sha==frozen_native_manifest_digest(raw338,valid)
            and not port.output_gate and not port.candidate_accepted,
            "original real 43M word338 and frozen quote sidecar immutable")
    tensors={
        "original_real43m_gt_word339_READONLY":extended.detach().contiguous(),
        "original_native_word_valid_mask":valid.detach().contiguous(),
        "native_BPE_causally_unsafe_word_mask":unsafe.detach().contiguous(),
        "source_row_indices_in_frozen_S74_NEW_TRAIN120":
            torch.arange(120,dtype=torch.int32),
    }
    report={
        "stage":STAGE,
        "original_protected_616_real_J7_replayed_before_new_S74_capture":True,
        "S74_SOURCE_ONLY_SHA256":source_sha256(),
        "no_S74_construction_labels_imported_or_used_during_any_native_H8":True,
        "no_S66_spent32_gold_EVAL32_EVAL128_imported":True,
        "genuine_43m_native_BPE_token_IDS_unchanged":True,
        "all_12_original_layers_and_original_GT_prefill_S93_profile":True,
        "original_model_weights_original_GT_expansion_adapters_untouched":True,
        "source_only_original_BPE_fractional_word_pool_no_gold":True,
        "n_new_S74_TRAIN_source_rows":120,
        "n_S74_grouped_train_source":sum(r["family"] in GROUP_TRAIN for r in rows),
        "n_S74_grouped_old_dev_source":sum(r["family"] in GROUP_DIAG for r in rows),
        "n_genuine_original_native_tokens":n_native_tokens,
        "max_raw_source_words":max_words,
        "n_causally_unsafe_shared_BPE_words":n_unsafe_words,
        "n_shared_BPE_native_pieces":n_shared_pieces,
        "n_original_quote_marks":int(quote_sidecar.text_only_quote_mask.sum()),
        "n_new_model_training_steps":0,
        "original_full_source_S95_and_S132_prefill_only_NOT_online_causal_safety":True,
        "no_candidate_quality_or_new_accuracy_measured":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    return tensors,report


def physical_source_only_commit(out:Path,tensors,report):
    require(out.exists() and out.is_dir()
            and not (out/TENSOR_NAME).exists()
            and not (out/REPORT_NAME).exists()
            and report["n_new_S74_TRAIN_source_rows"]==120
            and report["n_new_model_training_steps"]==0
            and report["output_gate"]==0
            and not report["candidate_accepted"]
            and not report["activation_allowed"]
            and report["original_protected_616_real_J7_replayed_before_new_S74_capture"],
            "fail-closed physical S75 source file output gate")
    import json
    p=out/TENSOR_NAME
    save_file(tensors,str(p))
    saved=load_file(str(p),device="cpu")
    require(set(saved)==set(tensors)
            and all(torch.equal(saved[k],tensors[k]) for k in tensors),
            "original FP16 native H8 and literal source offsets saved without changes")
    report["physical_source_tensor_SHA256"]=sha256(p.read_bytes()).hexdigest()
    (out/REPORT_NAME).write_text(json.dumps(
        report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf-8")
    return p
