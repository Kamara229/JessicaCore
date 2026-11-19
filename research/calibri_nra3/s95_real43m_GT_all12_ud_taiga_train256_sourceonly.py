"""NRA3 S95: read 256 TRUE Russian UD-Taiga TRAIN-only raw sentences
through frozen genuine 12-layer Calibri+GT after protected original616 J7.
NRA3's 22,311 new neural weights are installed INSIDE the research model
under model.model.nra3_neural_weights and left UNTRAINED/gate0.
Only the precommitted S94 source-only JSON is downloaded, never the
separate S94 word-index UD-GOLD labels. No old S79/S86 SPENT firstread
sources, no external inference parser, no old model training or decoder
changes. Original CB59 compressed physical SHA before/after required.
""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file,save_file

from calibri2.s160 import real_parallel_slots_j7 as s160
from calibri2.s8.broad_replay import SOURCE_PARAMS,PAID_OPERATORS,MODEL_SPEC
from v176.operators import clear
from research.calibri_nra3.s91_neural_multilayer_event_graph import (
    NRA3InCalibriNeuralWeights,STAGE as S91_STAGE,N_LAYERS,HIDDEN,
)
from research.nra3_s5.frozen_layer8_read_port import FrozenLayer8ReadPort
from research.nra3_s20.s75_original43m_gt_newtrain120_native_h8 import (
    source_offset_alignment,
)


STAGE="CALIBRI_NRA3_S95_REAL43M_GT_ALL12_NATIVE_UD_TAIGA_TRAIN256_POST_ORIGINAL616"
SOURCE_SHA="257fcc7518debb81e779c1419c34fdd9e565ad7e5ed9b04d08e55484fddba917"
S94_UPSTREAM_COMMIT="445ea57e47a3adae3146d0a42d2a8876214f2d20"
S94_LABEL_SHA="f6717e81c4620e1da0744b6aaaf3a83647ad76c7ebeeb42722195b432b4f1880"
N=256
TENSOR="NRA3_S95_REAL43M_GT_NATIVE_ALL12_LAYER320_UD_TRAIN256_READONLY.safetensors"
WEIGHTS="NRA3_S95_IN_CALIBRI_22311_NEW_NEURAL_WEIGHTS_FP16.safetensors"
REPORT="NRA3_S95_REAL43M_GT_ALL12_UD_TRAIN256_SOURCE_PROVENANCE.json"

def read_S94_SOURCE_only_not_labels():
    root=Path(os.environ["NRA3_S95_S94_SOURCE_ONLY_DIR"])
    # Entire S94 artifact has a separate label JSON, but this model
    # input reader must NEVER open/parse that path, or import gold.
    path=root/"NRA3_S94_TAIGA_TRAIN256_SOURCE_ONLY_NO_GOLD.json"
    manifest_path=root/"NRA3_S94_TAIGA_TRAIN256_LICENSE_PROVENANCE.json"
    require(path.is_file() and manifest_path.is_file(),
            "S94 new true UD Taiga source and provenance must be present")
    rows=json.loads(path.read_text(encoding="utf8"))
    manifest=json.loads(manifest_path.read_text(encoding="utf8"))
    h=hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True,
        separators=(",",":")).encode("utf8")).hexdigest()
    require(len(rows)==N and h==SOURCE_SHA and
            all(set(z)=={"row_id","text"} for z in rows) and
            all(z["row_id"]==f"S94_TAIGA_TRAIN_{i+1:03d}"
                and isinstance(z["text"],str)
                for i,z in enumerate(rows)) and
            len({z["text"] for z in rows})==N,
            "only locked source-only text rows, no label/role/UD parser input")
    require(manifest["source_only_sha256"]==SOURCE_SHA
            and manifest["separately_supervised_only_gold_sha256"]==S94_LABEL_SHA
            and manifest["upstream_immutable_git_commit"]==S94_UPSTREAM_COMMIT
            and manifest["n_training_rows"]==N
            and manifest["only_upstream_TRAIN_a_used_NO_upstream_dev_or_test"]
            and manifest["no_production_or_commercial_use_until_license_review"]
            and manifest["source_rows_have_no_label_or_UD_POS_or_role"]
            and manifest["new_NRA3_training_steps"]==0
            and not manifest["candidate_accepted"],
            "S94 original third-party TRAIN-only CC-BY-SA provenance must be fixed")
    return rows


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


def real_all12_source_capture_AFTER_original_616(
        model,sources,tok,bridge,tunnel,expansion):
    rows=read_S94_SOURCE_only_not_labels()
    require(len(rows)==N and all(set(z)=={"row_id","text"} for z in rows),
            "S94 real UD TRAIN256 source-only; old spent S79/S86 tests excluded")
    require(len(model.model.layers)==N_LAYERS and not model.training
            and next(model.parameters()).device.type=="cpu"
            and not expansion._active and not tunnel._active
            and not torch.cuda.is_available(),
            "real original model must be inactive 12layer CPU after616 J7")
    require(isinstance(sources,dict) and len(sources)==PAID_OPERATORS
            and "model.embed_tokens.weight" in sources
            and "model.layers.0.self_attn.q_proj.weight" in sources,
            "CB59 original 85 physical operator sources must remain full and frozen")
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
    require(len(examples)==N and layer8.observed_calls==N,
            "all 256 genuinely licensed Taiga UD TRAIN-only BPE prefills")
    last_digest,last_params=old_calibri_parameter_sha(model)
    require(old_digest==last_digest and old_params==last_params,
            "ALL original Calibri43M neural weights physically bit-identical after attaching NRA3")
    width=max(len(ids) for ids,_ in examples)
    native_hidden=torch.zeros((N,N_LAYERS,width,HIDDEN),dtype=torch.float16)
    native_ids=torch.full((N,width),-1,dtype=torch.int32)
    valid=torch.zeros((N,width),dtype=torch.bool)
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
        "source_row_indices_in_PRECOMMITTED_S94_UD_TRAIN256":
            torch.arange(N,dtype=torch.int32),
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
        "S94_REAL_UD_TAIGA_TRAIN_ONLY_SOURCE_SHA256":SOURCE_SHA,
        "S94_SEPARATE_OFFLINE_GOLD_SHA256_LOCKED_NOT_OPENED":S94_LABEL_SHA,
        "S94_licensed_CC_BY_SA_40_no_production_approval":True,
        "S94_UD_training_source_only_labels_NEVER_OPENED":True,
        "original_616_J7_physically_replayed_before_any_NRA3_hook":True,
        "genuine_original43M_CPU_12_layers_and_GreenTunnel_S93":True,
        "n_new_train_only_sources":N,
        "n_genuine_original_BPE_tokens":total_tokens,
        "max_native_BPE_tokens_per_source":width,
        "original_source_fractional_BPE_unsafe_raw_words":offset_unsafe,
        "original_layer8_S5_read_port_EQ_NRA3_all12_layer8_all72":True,
        "all_old_Calibri43M_weights_unchanged_sha256":old_digest,
        "n_original_compressed_CB59_parameter_elements_bit_identical":old_params,
        "original_S8_logical_core_parameter_count":SOURCE_PARAMS,
        "original_S8_paid_operator_count":PAID_OPERATORS,
        "original_S8_architecture":list(MODEL_SPEC),
        "NRA3_nnModule_REGISTERED_INSIDE_model_model":True,
        "NRA3_all_22311_new_params_belong_to_original_model_state_dict":True,
        "NRA2_8_neural_microareas_are_trainable_inside_Calibri":True,
        "learned_4_stream_router_over_ALL12_layers":True,
        "event_argument_neural_role_arcs_and_memory_are_in_weights":True,
        "S94_OFFLINE_UD_GOLD_not_imported_in_original_source_H8":True,
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
            and report["S94_OFFLINE_UD_GOLD_not_imported_in_original_source_H8"]
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
