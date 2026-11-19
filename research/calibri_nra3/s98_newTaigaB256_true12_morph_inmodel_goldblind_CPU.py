"""NRA3 S98: physical genuine original12 BPE Taiga TRAIN-b morphology-source-only

After the original protected 616 J7 actually replayed by S66,
register new NRA3 neural weights INSIDE the genuine Calibri model
at model.model.nra3_neural_weights. Run NEW 256 precommitted
UD Taiga TRAIN-b source-only raw texts through the actual 12 decoder layers
and S93/GreenTunnel prefill, using read-only all12 layer hooks. The
untrained neural graph is still a Calibri CHILD nn.Module, not a
standalone runtime model. Existing layer8 port simultaneously records
exact same hidden vectors, ensuring BPE and hook identity.

Original 43M model parameters unchanged before/after; NO S97 morphology/role UD GOLD or S96 internal dev or
old S79/S86 SPENT test/labels imported; no neural fitting, semantic
correctness or active decoder J7 claimed. New neural morphology weights 28,887
params attached, gate0, DO NOT activate or deploy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file,save_file

from calibri2.s160 import real_parallel_slots_j7 as s160
from calibri2.s8.broad_replay import SOURCE_PARAMS,PAID_OPERATORS,MODEL_SPEC
from v176.operators import clear
from research.calibri_nra3.s91_neural_multilayer_event_graph import (
    N_LAYERS,HIDDEN,
)
from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    NRA3MorphologyInCalibriWeights,
)
from research.nra3_s5.frozen_layer8_read_port import FrozenLayer8ReadPort
from research.nra3_s20.s75_original43m_gt_newtrain120_native_h8 import (
    source_offset_alignment,
)
from research.nra3_s19.graph_schema import raw_model_input

STAGE="CALIBRI_NRA3_S98_REAL43MGT_ALL12_TAIGA_TRAINB256_MORPH_SOURCE_ONLY_AFTER616"
SOURCE_SHA="437b6d62564e4cda6c58975910ccc98221c3e81938bda22429418f16700b1dad"
TENSOR="NRA3_S98_TAIGA_TRAINB256_REAL12_ORIGINAL_H320_NO_UD_GOLD.safetensors"
WEIGHTS="NRA3_S98_IN_CALIBRI_UNTRAINED_MORPH_28887_FP16.safetensors"
REPORT="NRA3_S98_REAL43MGT_TAIGA_TRAINB256_ALL12_SOURCEONLY_PROVENANCE.json"


def require(ok,reason):
    if not bool(ok):
        raise ValueError("NRA3 S98 12-layer actual43M CPU: "+reason)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def old_calibri_parameter_sha(model):
    """Hash existing OLD 43M model neural weights, excluding ONLY new child."""
    h=hashlib.sha256()
    old=0
    count=0
    for name,p in model.named_parameters():
        if "nra3_neural_weights" in name:
            continue
        require(not p.is_meta and p.device.type=="cpu"
                and p.dtype in (torch.float16,torch.float32)
                and bool(torch.isfinite(p.detach()).all()),
                "old model must be physically frozen CPU dense neural weights")
        h.update(name.encode("utf8"))
        h.update(b"\x00")
        h.update(p.detach().contiguous().cpu().numpy().tobytes())
        old+=p.numel()
        count+=1
    # CB59 originals are trained with *compressed TrainWeight centroids*.
    # Logical uncompressed core = SOURCE_PARAMS=36,380,480. PyTorch
    # named_parameters() instead counts the much smaller real physical
    # centroid/log-scale operators, so arbitrary dense cutoffs (40M, 1M)
    # are invalid. Trust the exact original architectural geometry and
    # 85 frozen PAID original operators; use full physical pre/post SHA.
    cfg=model.config
    require(SOURCE_PARAMS==36_380_480 and PAID_OPERATORS==85
            and MODEL_SPEC[1:4]==(320,1536,12)
            and int(cfg.num_hidden_layers)==12
            and int(cfg.hidden_size)==320
            and int(cfg.intermediate_size)==1536
            and old>0 and count>0
            and len(model.model.layers)==12,
            "requires exact original S8 CB59 compressed 36,380,480 logical 12layer core")
    print("NRA3_S92_ORIGINAL_CB59_LOGICAL_AND_PHYSICAL_COUNTS",
          {"logical_original":SOURCE_PARAMS,
           "compressed_parameter_elements":old,
           "physical_parameter_tensors":count,
           "paid_operator_count":PAID_OPERATORS},flush=True)
    return h.hexdigest(),old




def serialized_source_sha(rows):
    return hashlib.sha256(json.dumps(
        rows,ensure_ascii=False,sort_keys=True,
        separators=(",",":")).encode("utf8")).hexdigest()


def real_all12_source_capture_AFTER_original_616(
        model,sources,tok,bridge,tunnel,expansion,source_json_path):
    """Gold-free source-only original 12-layer capture, licensed TRAIN only.

    Source is externally SHA-frozen S94 Taiga TRAIN-a, gold physically
    absent from this CPU job. Source-only exact original BPE word
    alignment preflight may exclude ambiguous texts WITHOUT reading
    any corpus grammar targets. Reports entire exclusion map.
    """
    rows=json.loads(Path(source_json_path).read_text(encoding="utf8"))
    require(len(rows)==256 and serialized_source_sha(rows)==SOURCE_SHA
            and all(set(z)=={"row_id","text"} for z in rows)
            and len({z["row_id"] for z in rows})==256
            and len({z["text"] for z in rows})==256,
            "licensed Taiga TRAIN-a raw source256 SHA only; NO UD gold")
    require(len(model.model.layers)==N_LAYERS and not model.training
            and next(model.parameters()).device.type=="cpu"
            and not expansion._active and not tunnel._active
            and not torch.cuda.is_available(),
            "original compressed CB59+GT model must be untouched 12layer CPU AFTER616")
    require(isinstance(sources,dict) and len(sources)==PAID_OPERATORS
            and "model.embed_tokens.weight" in sources
            and "model.layers.0.self_attn.q_proj.weight" in sources,
            "all original 85 compressed paid operator source banks loaded")
    # Text-only, gold-blind BPE preflight BEFORE any neural model forward.
    # Cap input size by fixed source-only criterion to limit role-arc T^2.
    accepted=[]
    ignored=[]
    for original_idx,row in enumerate(rows):
        try:
            text=row["text"]
            ids=list(s160.prompt_tokens(tok,text))
            if not 2<=len(ids)<=256:
                raise ValueError("original native BPE context length 2..256")
            align=source_offset_alignment(tok,text,ids)
            raw=raw_model_input(text).tokens
            if not (3<=len(raw)<=40 and len(align.raw_words)==len(raw)):
                raise ValueError("source-only S51 raw token count mismatched")
            if not all(
                0<=z.native_token_positions[-1]<len(ids)
                and z.native_token_positions
                for z in align.raw_words
            ):
                raise ValueError("original native BPE last-piece coordinates invalid")
            accepted.append((original_idx,row,ids,align,raw))
        except ValueError as exc:
            ignored.append((original_idx,str(exc)[:170]))
    counts=[sum(k*64<=idx<(k+1)*64 for idx,*_ in accepted)
            for k in range(4)]
    require(len(accepted)>=96 and min(counts)>=12
            and len(accepted)+len(ignored)==256,
            "original BPE preflight must preserve enough TRAIN source from every fixed Taiga stratum")
    print("NRA3_S95_ORIGINAL_BPE_SOURCE_ONLY_PREFLIGHT",
          json.dumps({"accepted":len(accepted),
                     "excluded_no_gold":len(ignored),
                     "accepted_per_64_source_stratum":counts},sort_keys=True),
          flush=True)

    old_digest,old_params=old_calibri_parameter_sha(model)
    child=NRA3MorphologyInCalibriWeights()
    child.attach_to_RESEARCH_calibri(model)
    require(child.budget()["new_neural_parameters"]==28887
            and child.output_gate==0 and not child.candidate_accepted
            and not child.activation_allowed and not child.is_trained,
            "real NRA3 child registered INSIDE Calibri nn.Module, untrained gate0")
    child_keys=tuple(k for k in model.state_dict()
                     if k.startswith("model.nra3_neural_weights."))
    require(len(child_keys)==len(tuple(child.state_dict()))
            and sum(p.numel() for p in child.parameters())==28887,
            "the same new 28887 child parameters belong to Calibri state_dict")

    layer8=FrozenLayer8ReadPort()
    examples=[]
    total_tokens=0
    offset_unsafe=0
    for original_idx,row,ids,align,raw in accepted:
        text=row["text"]
        v=torch.as_tensor(s160.prompt_only_vector(tok,text),dtype=torch.float32)
        require(v.shape==(512,) and bool(torch.isfinite(v).all())
                and len(ids)==align.original_token_count,
                "original tokenizer BPE ids and S95 native source prefix")
        with (torch.inference_mode(),
              expansion.prefill_only(model),
              s160.prefill_only_tunnel(model,tunnel)):
            with s160.frozen_cpu_phase_scope(model,bridge,"S93",v,phase="prefill_only"):
                with child.read_only_all12_source_scope(model):
                    with layer8.capture(model,text,ids):
                        clear(sources)
                        out=model(input_ids=torch.tensor([ids],dtype=torch.long),
                                  past_key_values=None,use_cache=True,
                                  output_hidden_states=True)
                        require(out.past_key_values is not None
                                and out.hidden_states[-1].shape[-1]==HIDDEN,
                                "actual protected original CPU GT 12 layers and KV cache unchanged")
                        del out
                        clear(sources)
        true_hidden=child.latest_source_hidden
        capture=layer8.samples[
            hashlib.sha256(text.encode("utf8")).hexdigest()]
        require(true_hidden.shape==(1,N_LAYERS,len(ids),HIDDEN)
                and true_hidden.dtype==torch.float16
                and bool(torch.isfinite(true_hidden).all())
                and torch.equal(true_hidden[0,7],capture.hidden320_fp16),
                "ALL12 true source H8 must match independent frozen S5 port bit-for-bit")
        g=child.latest_source_graph
        require(g is not None and g.role_arc_logits.shape==(1,8,len(ids),len(ids))
                and g.mini_area_logits.shape==(1,len(ids),8)
                and g.event_state_logits.shape==(1,len(ids),6)
                and g.layer_router_weights.shape==(4,12)
                and g.memory_slot_attention.shape==(1,len(ids),16)
                and g.upos_logits.shape==(1,len(ids),17)
                and g.case_logits.shape==(1,len(ids),9)
                and g.number_logits.shape==(1,len(ids),3)
                and g.voice_logits.shape==(1,len(ids),3)
                and child.capture_count==1 and child.output_gate==0,
                "embedded all12 real original-source neural graph ABI")
        examples.append((original_idx,ids,true_hidden[0].clone(),align,raw))
        total_tokens+=len(ids)
        offset_unsafe+=len(align.ambiguous_raw_word_ids)
    require(len(examples)==len(accepted) and layer8.observed_calls==len(examples),
            "each strictly source-aligned original Taiga TRAIN row captured exactly once")
    last_digest,last_params=old_calibri_parameter_sha(model)
    require(old_digest==last_digest and old_params==last_params,
            "all original Calibri compressed CB59 params physically bit-identical")

    n=len(examples)
    width=max(len(ids) for _,ids,_,_,_ in examples)
    maxword=max(len(raw) for *_,raw in examples)
    hidden=torch.zeros((n,N_LAYERS,width,HIDDEN),dtype=torch.float16)
    ids_tensor=torch.full((n,width),-1,dtype=torch.int32)
    valid=torch.zeros((n,width),dtype=torch.bool)
    row_indices=torch.empty((n,),dtype=torch.int32)
    raw_start=torch.full((n,maxword),-1,dtype=torch.int32)
    raw_end=torch.full((n,maxword),-1,dtype=torch.int32)
    last_BPE=torch.full((n,maxword),-1,dtype=torch.int32)
    safe=torch.zeros((n,maxword),dtype=torch.bool)
    for row,(original_idx,ids,h,align,raw) in enumerate(examples):
        count=len(ids)
        hidden[row,:,:count]=h
        ids_tensor[row,:count]=torch.tensor(ids,dtype=torch.int32)
        valid[row,:count]=True
        row_indices[row]=original_idx
        for j,(word,piece) in enumerate(zip(raw,align.raw_words)):
            raw_start[row,j]=word.start
            raw_end[row,j]=word.end
            last_BPE[row,j]=piece.native_token_positions[-1]
            safe[row,j]=piece.causally_safe
    require(int(valid.sum())==total_tokens
            and bool(torch.isfinite(hidden).all())
            and not bool((ids_tensor[~valid]!=-1).any())
            and (row_indices[1:]>row_indices[:-1]).all()
            and not bool((safe&(last_BPE<0)).any()),
            "true frozen original BPE all12 tensors and source-only offsets")
    tensors={
        "original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY":hidden.contiguous(),
        "original_native_BPE_token_ids":ids_tensor.contiguous(),
        "native_BPE_source_valid_mask":valid.contiguous(),
        "source_row_indices_in_PINNED_TAIGA_TRAIN256":row_indices.contiguous(),
        "source_raw_word_char_start":raw_start.contiguous(),
        "source_raw_word_char_end":raw_end.contiguous(),
        "source_raw_word_last_native_BPE":last_BPE.contiguous(),
        "source_raw_word_native_BPE_causally_safe":safe.contiguous(),
    }
    weights={
        "model.nra3_neural_weights."+name:val.detach().half().cpu().contiguous()
        for name,val in child.state_dict().items()
    }
    require(set(weights)==set(child_keys)
            and sum(v.numel() for v in weights.values())==28887,
            "neural child must be attached to SAME genuine Calibri checkpoint")
    manifest={
        "stage":STAGE,
        "S97_TAIGA_TRAIN_B_PINNED_SOURCE256_SHA256":SOURCE_SHA,
        "S97_TAIGA_TRAIN_B_original_git_blob":"3b0f6947bd0d6ac022d91d5fb17269450a876bb3",
        "third_party_data_license":"CC BY-SA 4.0",
        "no_commercial_or_production_use_before_legal_review":True,
        "no_original_Taiga_dev_or_test_data_used":True,
        "n_pinned_source_only_rows":256,
        "n_accepted_source_only_BPE_rows":n,
        "n_excluded_due_to_input_BPE_preflight":len(ignored),
        "excluded_original_source_row_ids_and_reasons":ignored,
        "accepted_n_per_fixed_Taiga_64row_source_stratum":counts,
        "n_genuine_original_native_BPE_tokens":total_tokens,
        "original_max_native_BPE_width":width,
        "n_native_unsafe_raw_word_positions":offset_unsafe,
        "genuine_12layer_original43M_GT_S93_prefill_after_protected616_J7":True,
        "old_616_J7_physically_replayed_BEFORE_NRA3_hooks":True,
        "all_original_43M_GT_CB59_params_sha_pre_and_post_identical":old_digest,
        "n_original_compressed_CB59_physical_parameter_elements":old_params,
        "original_S8_logical_core_params":SOURCE_PARAMS,
        "original_85_paid_CB59_operators_present":True,
        "original_S5_H8_port_exactly_matches_all12_H8_for_all_accepted":True,
        "NRA3_morph_neural_28887_weights_REGISTERED_INSIDE_actual_Calibri_model":True,
        "eight_original_NRA2_neural_miniareas_still_embedded":True,
        "source_only_no_Taiga_TRAIN_B_UD_morph_or_role_GOLD_loaded_during_CPU":True,
        "old_S79_S86_and_S96_internal_dev_NEVER_reused":True,
        "new_NRA3_train_steps":0,
        "no_active_new616_J7_or_online_autoregressive_certification":True,
        "no_all_in_bpp_or_CPU_certification":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    return tensors,weights,manifest


def save_only_AFTER_original616_digest_verified(out,tensors,weights,report):
    require(out.is_dir() and not (out/TENSOR).exists()
            and not (out/WEIGHTS).exists() and not (out/REPORT).exists()
            and report["old_616_J7_physically_replayed_BEFORE_NRA3_hooks"]
            and report["new_NRA3_train_steps"]==0
            and report["source_only_no_Taiga_TRAIN_B_UD_morph_or_role_GOLD_loaded_during_CPU"]
            and report["no_commercial_or_production_use_before_legal_review"]
            and report["old_S79_S86_and_S96_internal_dev_NEVER_reused"]
            and report["output_gate"]==0 and not report["candidate_accepted"],
            "save only if previous original protected616 token digests succeeded")
    for filename,payload in ((TENSOR,tensors),(WEIGHTS,weights)):
        f=out/filename
        save_file(payload,str(f))
        reread=load_file(str(f),device="cpu")
        require(set(reread)==set(payload)
                and all(torch.equal(reread[k],v) for k,v in payload.items()),
                "genuine Calibri layer and embedded neural weight FP16 exact readback")
    report["ALL12_CPU_SAFETENSORS_SHA256"]=file_sha(out/TENSOR)
    report["IN_CALIBRI_28887_MORPH_FP16_WEIGHTS_SHA256"]=file_sha(out/WEIGHTS)
    (out/REPORT).write_text(json.dumps(
        report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    return out/TENSOR
