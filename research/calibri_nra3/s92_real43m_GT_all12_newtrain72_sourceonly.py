"""NRA3 S92: physical first genuine original43M GT all12 CPU TRAIN source.

After the original protected 616 J7 actually replayed by S66,
register new NRA3 neural weights INSIDE the genuine Calibri model
at model.model.nra3_neural_weights. Run 72 previously precommitted
S83 TRAIN-only raw source texts through the actual 12 decoder layers
and S93/GreenTunnel prefill, using read-only all12 layer hooks. The
untrained neural graph is still a Calibri CHILD nn.Module, not a
standalone runtime model. Existing layer8 port simultaneously records
exact same hidden vectors, ensuring BPE and hook identity.

Original 43M model parameters unchanged before/after; NO S83 GOLD or
old S79/S86 SPENT test/labels imported; no neural fitting, semantic
correctness or active decoder J7 claimed. New neural weights 22,311
params attached, gate0, DO NOT activate or deploy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file,save_file

from calibri2.s160 import real_parallel_slots_j7 as s160
from v176.operators import clear
from research.calibri_nra3.s91_neural_multilayer_event_graph import (
    NRA3InCalibriNeuralWeights,STAGE as S91_STAGE,N_LAYERS,HIDDEN,
)
from research.nra3_s5.frozen_layer8_read_port import FrozenLayer8ReadPort
from research.nra3_s20.s75_original43m_gt_newtrain120_native_h8 import (
    source_offset_alignment,
)
from research.nra3_s20.s83_NEW_DIVERSE72_source_ONLY_never_use_S79_spent import (
    source_only_new72,source_only_new72_sha256,
)

STAGE="CALIBRI_NRA3_S92_REAL43M_GT_ALL12_NATIVE_TRAIN72_POST_ORIGINAL616"
SOURCE_SHA="285475efb25a1310ce87fa5ffc9e8ec99dd5f129d147b0edf4afa7195f43a62a"
TENSOR="NRA3_S92_REAL43M_GT_NATIVE_ALL12_LAYER320_TRAIN72_READONLY.safetensors"
WEIGHTS="NRA3_S92_IN_CALIBRI_22311_NEW_NEURAL_WEIGHTS_FP16.safetensors"
REPORT="NRA3_S92_REAL43M_GT_ALL12_TRAIN72_SOURCE_PROVENANCE.json"


def require(ok,reason):
    if not bool(ok):
        raise ValueError("NRA3 S92 12-layer actual43M CPU: "+reason)


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
    # The published ~43M Calibri+GreenTunnel experiment spans separately
    # loaded banks, bridges and checkpoints; the HF core nn.Module alone
    # need not contain 40M params. Compare exact full OLD core digest
    # before/after attachment instead of inventing a core-only count.
    require(old>=1_000_000 and count>=12,
            "expected populated genuine 12layer original frozen neural core")
    print("NRA3_S92_ACTUAL_FROZEN_BASE_CORE_PARAMETER_COUNT",old,
          "N_PARAM_TENSORS",count,flush=True)
    return h.hexdigest(),old


def real_all12_source_capture_AFTER_original_616(
        model,sources,tok,bridge,tunnel,expansion):
    rows=source_only_new72()
    require(len(rows)==72 and source_only_new72_sha256()==SOURCE_SHA
            and all(set(z)=={"row_id","group","text"} for z in rows),
            "S83 NEW TRAIN72 source only precommitted, never S79/S86 tests")
    require(len(model.model.layers)==N_LAYERS and not model.training
            and next(model.parameters()).device.type=="cpu"
            and not expansion._active and not tunnel._active
            and not torch.cuda.is_available(),
            "real original model must be inactive 12layer CPU after616 J7")
    old_digest,old_params=old_calibri_parameter_sha(model)
    child=NRA3InCalibriNeuralWeights()
    child.attach_to_RESEARCH_calibri(model)
    require(child.budget()["new_neural_parameters"]==22311
            and child.output_gate==0 and not child.candidate_accepted
            and not child.activation_allowed and not child.is_trained,
            "NRA3 must be real Calibri child weights, NEVER active")
    child_keys=tuple(k for k in model.state_dict()
                     if k.startswith("model.nra3_neural_weights."))
    require(len(child_keys)==len(tuple(child.state_dict()))
            and sum(p.numel() for p in child.parameters())==22311,
            "child state_dict genuinely included inside Calibri")
    layer8=FrozenLayer8ReadPort()
    examples=[]
    total_tokens=0
    offset_unsafe=0
    for row in rows:
        text=row["text"]
        ids=list(s160.prompt_tokens(tok,text))
        align=source_offset_alignment(tok,text,ids)
        v=torch.as_tensor(s160.prompt_only_vector(tok,text),dtype=torch.float32)
        require(v.shape==(512,) and bool(torch.isfinite(v).all())
                and len(ids)>=2
                and align.original_token_count==len(ids),
                "exact original native token IDs and S95 source prefill")
        with (torch.inference_mode(),
              expansion.prefill_only(model),
              s160.prefill_only_tunnel(model,tunnel)):
            with s160.frozen_cpu_phase_scope(
                    model,bridge,"S93",v,phase="prefill_only"):
                with child.read_only_all12_source_scope(model):
                    with layer8.capture(model,text,ids):
                        clear(sources)
                        output=model(
                            input_ids=torch.tensor([ids],dtype=torch.long),
                            past_key_values=None,use_cache=True,
                            output_hidden_states=True)
                        require(output.past_key_values is not None
                                and output.hidden_states[-1].shape[-1]==HIDDEN,
                                "original full twelve CPU blocks and native cache unchanged")
                        del output
                        clear(sources)
        observed=child.latest_source_hidden
        captured=layer8.samples[
            hashlib.sha256(text.encode("utf8")).hexdigest()]
        require(observed.shape==(1,N_LAYERS,len(ids),HIDDEN)
                and observed.dtype==torch.float16
                and bool(torch.isfinite(observed).all())
                and torch.equal(observed[0,7],captured.hidden320_fp16),
                "NRA3 real ALL12 original H8 port IDENTICAL BPE hidden")
        graph=child.latest_source_graph
        require(graph is not None and graph.role_arc_logits.shape==
                    (1,8,len(ids),len(ids))
                and graph.mini_area_logits.shape==(1,len(ids),8)
                and graph.event_state_logits.shape==(1,len(ids),6)
                and graph.layer_router_weights.shape==(4,12)
                and graph.memory_slot_attention.shape==(1,len(ids),16)
                and graph.event_mention_logits.shape==(1,len(ids))
                and child.capture_count==1 and child.output_gate==0,
                "real embedded NRA3 neural role/mini-area graph valid on each original prefill")
        examples.append((ids,observed[0].clone()))
        total_tokens+=len(ids)
        offset_unsafe+=len(align.ambiguous_raw_word_ids)
    require(len(examples)==72 and layer8.observed_calls==72,
            "all 72 previously precommitted training-only source rows true CPU")
    last_digest,last_params=old_calibri_parameter_sha(model)
    require(old_digest==last_digest and old_params==last_params,
            "ALL original Calibri43M neural weights physically bit-identical after attaching NRA3")
    width=max(len(ids) for ids,_ in examples)
    native_hidden=torch.zeros((72,N_LAYERS,width,HIDDEN),dtype=torch.float16)
    native_ids=torch.full((72,width),-1,dtype=torch.int32)
    valid=torch.zeros((72,width),dtype=torch.bool)
    for row,(ids,h) in enumerate(examples):
        n=len(ids)
        native_hidden[row,:,:n]=h
        native_ids[row,:n]=torch.tensor(ids,dtype=torch.int32)
        valid[row,:n]=True
    require(int(valid.sum())==total_tokens
            and bool(torch.isfinite(native_hidden).all())
            and bool((native_hidden[~valid.unsqueeze(1).expand_as(native_hidden[:,:,:,0])]
                      ==0).all()),
            "all12 genuine original native tensor padding and finite")
    tensors={
        "original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY":
            native_hidden.contiguous(),
        "original_native_BPE_token_ids":native_ids.contiguous(),
        "native_BPE_source_valid_mask":valid.contiguous(),
        "source_row_indices_in_PRECOMMITTED_S83_TRAIN72":
            torch.arange(72,dtype=torch.int32),
    }
    weights={
        "model.nra3_neural_weights."+key:val.detach().half().cpu().contiguous()
        for key,val in child.state_dict().items()
    }
    require(sum(v.numel() for v in weights.values())==22311
            and set(weights)==set(child_keys),
            "new neural parameters MUST be same Calibri child state_dict keys")
    manifest={
        "stage":STAGE,
        "S83_PRECOMMITTED_TRAIN_SOURCE_SHA256":source_only_new72_sha256(),
        "original_616_J7_physically_replayed_before_any_NRA3_hook":True,
        "genuine_original43M_CPU_12_layers_and_GreenTunnel_S93":True,
        "n_new_train_only_sources":72,
        "n_genuine_original_BPE_tokens":total_tokens,
        "max_native_BPE_tokens_per_source":width,
        "original_source_fractional_BPE_unsafe_raw_words":offset_unsafe,
        "original_layer8_S5_read_port_EQ_NRA3_all12_layer8_all72":True,
        "all_old_Calibri43M_weights_unchanged_sha256":old_digest,
        "n_original_Calibri_params_bit_identical":old_params,
        "NRA3_nnModule_REGISTERED_INSIDE_model_model":True,
        "NRA3_all_22311_new_params_belong_to_original_model_state_dict":True,
        "NRA2_8_neural_microareas_are_trainable_inside_Calibri":True,
        "learned_4_stream_router_over_ALL12_layers":True,
        "event_argument_neural_role_arcs_and_memory_are_in_weights":True,
        "S83_CONSTRUCTED_GOLD_not_imported_in_original_source_H8":True,
        "old_S79_and_S86_spent_firstreads_NEVER_reused":True,
        "new_NRA3_training_steps":0,
        "new_integrated_616_J7_not_run":True,
        "online_autoregressive_causality_NOT_certified":True,
        "all_in_bpp_le07_and_CPU_ge90_NOT_certified":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    return tensors,weights,manifest


def save_only_AFTER_original616_digest_verified(out,tensors,weights,report):
    require(out.is_dir() and not (out/TENSOR).exists()
            and not (out/WEIGHTS).exists() and not (out/REPORT).exists()
            and report["original_616_J7_physically_replayed_before_any_NRA3_hook"]
            and report["new_NRA3_training_steps"]==0
            and report["S83_CONSTRUCTED_GOLD_not_imported_in_original_source_H8"]
            and report["old_S79_and_S86_spent_firstreads_NEVER_reused"]
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
    report["IN_CALIBRI_22311_NEW_FP16_WEIGHTS_SHA256"]=file_sha(out/WEIGHTS)
    (out/REPORT).write_text(json.dumps(
        report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    return out/TENSOR
