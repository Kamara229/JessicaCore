"""NRA2 S84: FIRST physical original Calibri43M+GreenTunnel H8 read
of S83 SHA-locked NEW, TRAIN-only 72 newly authored diverse Russian sentences.

HARD WALL: this module does NOT import S83 constructed target annotations
or S82 model/weights, has no event labels, no target classes, no fitting
or test-driven selection. S83 source/label Git blobs + precommit workflow
#44514452404 already SUCCESS BEFORE any genuine model read.

The original S66 616 protected J7 CPU replay precedes this physical
capture. The wrapper persists output ONLY after unchanged posthoc
original baseline digest validation. No main/live/original changes.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import torch
from safetensors.torch import save_file,load_file

from calibri2.s160 import real_parallel_slots_j7 as s160
from v176.operators import clear
from research.nra3_s5.frozen_layer8_read_port import FrozenLayer8ReadPort
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s49_real43m_gt_quote_sidecar_zero_gate import (
    Frozen43MGreenTunnelQuotePort,frozen_native_manifest_digest,
)
from research.nra3_s20.s51_fractional_native_bpe_overlap import (
    fractional_native_word_pool,
)
from research.nra3_s20.s75_original43m_gt_newtrain120_native_h8 import (
    source_offset_alignment,
)
from research.nra3_s20.s83_NEW_DIVERSE72_source_ONLY_never_use_S79_spent import (
    source_only_new72,source_only_new72_sha256,
)

EXPECTED_SOURCE_SHA="285475efb25a1310ce87fa5ffc9e8ec99dd5f129d147b0edf4afa7195f43a62a"
STAGE="NRA2_S84_TRUE43M_GT_FIRSTREAD_DIVERSE72_GOLD_BLIND_H8_SOURCE"
TENSOR_NAME="NRA2_S84_GENUINE43M_GT_DIVERSE72_WORD339_NO_GOLD.safetensors"
REPORT_NAME="NRA2_S84_GENUINE43M_GT_DIVERSE72_GOLD_BLIND_SOURCE_PROVENANCE.json"


def require(ok,why):
    if not bool(ok):
        raise ValueError("NRA2 S84 original43M FIRST-read source-only: "+why)


def capture_new_diverse72_original43m_GT_native_only(model,sources,tok,
                                                   bridge,tunnel,expansion):
    texts=source_only_new72()
    require(len(texts)==72
            and source_only_new72_sha256()==EXPECTED_SOURCE_SHA
            and all(set(r)=={"row_id","group","text"} for r in texts),
            "pre-registered S83 NEW TRAIN-only 72 SOURCE-only identity SHA frozen")
    require(len(model.model.layers)==12 and not model.training
            and next(model.parameters()).device.type=="cpu"
            and not expansion._active and not tunnel._active
            and not torch.cuda.is_available(),
            "original 12-layer protected physical CPU no model mutation")
    collector=FrozenLayer8ReadPort()
    aligns=[]
    for row in texts:
        text=row["text"]
        ids=list(s160.prompt_tokens(tok,text))
        m=source_offset_alignment(tok,text,ids)
        vec=torch.as_tensor(s160.prompt_only_vector(tok,text),
                            dtype=torch.float32)
        require(vec.shape==(512,) and bool(torch.isfinite(vec).all()),
                "original S95 full source features as original model receives")
        with (torch.inference_mode(),
              expansion.prefill_only(model),
              s160.prefill_only_tunnel(model,tunnel)):
            with s160.frozen_cpu_phase_scope(
                    model,bridge,"S93",vec,phase="prefill_only"):
                with collector.capture(model,text,ids):
                    clear(sources)
                    y=model(input_ids=torch.tensor([ids],dtype=torch.long),
                            past_key_values=None,use_cache=True,
                            output_hidden_states=True)
                    require(y.past_key_values is not None
                            and y.hidden_states[-1].shape[-1]==320,
                            "real original 43M all12 source prefill H8")
                    del y
                    clear(sources)
        aligns.append(m)
    require(collector.observed_calls==72 and len(aligns)==72,
            "S84 each new source exactly once never model selection")
    count=[len(z.raw_words) for z in aligns]
    width=max(count)
    require(min(count)>=3 and width<=40,
            "original S49 raw Russian quote gate supports 3..40 words")
    raw338=torch.zeros((72,width,338),dtype=torch.float32)
    valid=torch.zeros((72,width),dtype=torch.bool)
    unsafe=torch.zeros((72,width),dtype=torch.bool)
    tokens=0;ambiguous=0;shared=0
    for i,(row,alignment) in enumerate(zip(texts,aligns)):
        cap=collector.samples[hashlib.sha256(
            row["text"].encode("utf-8")).hexdigest()]
        original=cap.hidden320_fp16
        require(tuple(cap.token_ids)==alignment.original_token_ids
                and original.dtype==torch.float16
                and original.shape==(alignment.original_token_count,320)
                and not original.requires_grad,
                "first-read genuine original BPE tokens and real H8")
        shadow=original.clone()
        pooled=fractional_native_word_pool(original,alignment)
        require(torch.equal(original,shadow)
                and pooled.shape==(count[i],320),
                "genuine original H8 and native tokenizer cannot mutate")
        words=raw_surface_grammar(row["text"])
        require(len(words)==count[i] and count[i]<=width,
                "original raw S18 grammar and source BPE offsets aligned")
        n=count[i]
        raw338[i,:n,:320]=pooled
        raw338[i,:n,320:338]=torch.tensor(
            [z.features for z in words],dtype=torch.float32)
        valid[i,:n]=True
        if alignment.ambiguous_raw_word_ids:
            unsafe[i,list(alignment.ambiguous_raw_word_ids)]=True
        tokens+=alignment.original_token_count
        ambiguous+=len(alignment.ambiguous_raw_word_ids)
        shared+=len(alignment.shared_native_token_ids)
    require(bool(torch.isfinite(raw338).all())
            and not bool((unsafe&~valid).any()),
            "S84 raw original vector finite and offset unsafe mask valid")
    sha_before=frozen_native_manifest_digest(raw338,valid)
    port=Frozen43MGreenTunnelQuotePort()
    raw_identity=port.forward_original_unchanged(
        raw338,valid,tuple(z["text"] for z in texts))
    with_quote,quote=port.shadow_augmented_features_only(
        raw338,valid,tuple(z["text"] for z in texts))
    require(raw_identity is raw338
            and raw_identity.data_ptr()==raw338.data_ptr()
            and sha_before==frozen_native_manifest_digest(raw338,valid)
            and with_quote.shape==(72,width,339)
            and torch.equal(with_quote[:,:,:338],raw338)
            and port.output_gate==0 and not port.candidate_accepted,
            "strict original Calibri H8 and read-only quote append identity")
    tensors={
        "original_real43m_gt_word339_READONLY":
            with_quote.detach().contiguous(),
        "original_native_word_valid_mask":valid.detach().contiguous(),
        "native_BPE_causally_unsafe_word_mask":unsafe.detach().contiguous(),
        "source_row_indices_in_frozen_S83_DIVERSE72":
            torch.arange(72,dtype=torch.int32),
    }
    report={
        "stage":STAGE,
        "original_Calibri43M_protected_616J7_physically_replayed_BEFORE_training_source":True,
        "S83_new_diverse72_SOURCE_ONLY_SHA256":source_only_new72_sha256(),
        "S83_PRECOMMIT_SUCCESS_WORKFLOW_RUN_ID":44514452404,
        "S83_CONSTRUCTION_GOLD_NEVER_IMPORTED_BY_REAL_H8_CPU":True,
        "S82_architecture_weights_NEVER_LOADED_BY_SOURCE_H8_CAPTURE":True,
        "original_43M_real_frozen_BPE_IDs_AND_offset_mapping_unchanged":True,
        "genuine_original_12_layers_with_GT_S132_prefill_S93":True,
        "original_S26_S57_S59_S61_S62_S63_S65_and_GT_weights_unchanged":True,
        "first_source_only_EVAL48_untrained_no_gold_and_no_classification":True,
        "n_diverse_train_only_input_rows":72,
        "n_original_native_BPE_tokens":tokens,
        "max_native_raw_source_words":width,
        "n_shared_native_BPE_unsafe_word_positions":ambiguous,
        "n_native_shared_token_pieces":shared,
        "n_quote_marked_original_raw_words":int(quote.text_only_quote_mask.sum()),
        "new_model_training_steps":0,
        "source_full_H8_not_online_causal_safety_certified":True,
        "no_scores_or_model_selection_before_S82_training":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    return tensors,report


def persist_source_only_AFTER_original_J7_verified(out:Path,tensors,report):
    require(out.is_dir() and not (out/TENSOR_NAME).exists()
            and not (out/REPORT_NAME).exists()
            and report["n_diverse_train_only_input_rows"]==72
            and report["S83_CONSTRUCTION_GOLD_NEVER_IMPORTED_BY_REAL_H8_CPU"]
            and report["S82_architecture_weights_NEVER_LOADED_BY_SOURCE_H8_CAPTURE"]
            and report["original_Calibri43M_protected_616J7_physically_replayed_BEFORE_training_source"]
            and report["output_gate"]==0
            and not report["candidate_accepted"]
            and not report["activation_allowed"],
            "S84 only after pristine original 616 J7 protected digests")
    path=out/TENSOR_NAME
    save_file(tensors,str(path))
    verify=load_file(str(path),device="cpu")
    require(set(verify)==set(tensors)
            and all(torch.equal(verify[k],tensors[k]) for k in tensors),
            "S84 genuine source-only FP16 exact bytes readback")
    report["genuine_source_tensor_file_SHA256"]=hashlib.sha256(
        path.read_bytes()).hexdigest()
    (out/REPORT_NAME).write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    return path
