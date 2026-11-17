#!/usr/bin/env python3
"""S7A REAL working 43.26M 12-layer CALIBRI -> FP32 mirror -> new CB61.

The FP32 mirror is *DECODED FROM* actual S93/S142 CALIBRI trained weights;
it is NOT an independently pretrained public FP32 43M teacher. Exact input
and output parity is expected only for this same-source diagnostic control.
New source-only K32V8 re-encoding is an actual different physical checkpoint.
All original S142 neural GT / Q2 / S95 / S98 / progressive source intact.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import struct
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59, encode
from calibri2.s8.broad_replay import MODEL_SPEC,SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s87.internal_green_tunnel import IntegratedGreenTunnel
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from calibri2.s78 import mps_36m_feature_cached_late_rank16_restoration as s78
from calibri2.s82 import real_mps_generic_token_context_copy as s82
from calibri2.s95.train_and_audit_router import CHOICES,prompt_only_vector
from calibri2.s99.guard_neural_copy_456_cpu import load_frozen_numeric_weights
from calibri2.s101.kv_cache_full456_cpu import cached_full_cpu_with_neural_copy
from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel
from calibri2.s112.prompt_conditioned_first_token_mps import load_cpu_original
from calibri2.s126.full_layer_state_tunnel import FullLayerStateTunnel
from calibri2.s126.compact_rank4_q2 import load_compact_q2,PREFIX
from calibri2.s127.q2_prefill_gate_control import prefill_only_tunnel
from calibri2.s132.progressive_core import (
    ProgressiveRouteCore, IDS_KEY, WEIGHTS_KEY, initial_active_indices
)
from v176.operators import clear

BASE_36M=36380480
EXTRA_43M=6881280
TOTAL_43M=43261760
SOURCE_CB61_SHA="18e5396e42fba01b4c85fc773f196c65c9badca5a7506f0923eab540813a92b7"
S142_SHA="5a706d9012a61f24678929a2a6ee20f77c922e7e0f4bb516c061f32742cc56c7"
S93_SIDECAR_SHA="093ef140c254a14c214fd7057a82d635e3f4275dae98d04cb365954a10aa37e0"
SELECTED_DEBUG=8
MAX_BPP=.7
HEADER=struct.Struct("<8sQI32s")
assert BASE_36M==SOURCE_PARAMS and BASE_36M+EXTRA_43M==TOTAL_43M


def digest(b):
    return hashlib.sha256(b).hexdigest()


def dump(p,obj):
    Path(p).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,
                indent=2)+"\n",encoding="utf-8")


def log(stage,**kwargs):
    print(json.dumps({"stage":stage,**kwargs},ensure_ascii=False),flush=True)


def new_CB61(raw):
    """Same already verified zlib CB61 format, WITHOUT another stored source."""
    packed=HEADER.pack(b"CB61ZL01",len(raw),zlib.crc32(raw),bytes.fromhex(digest(raw)))
    packed+=zlib.compress(raw,level=9)
    if s87.open_envelope(packed)!=raw:
        raise AssertionError("S7A 43M newly recompressed CB61 cannot be verified and decoded")
    return packed


def validate_parent(input):
    cb61=Path(input.S93_CB61).read_bytes()
    side_raw=Path(input.S93_sidecar).read_bytes()
    s142_raw=Path(input.S142_FP16).read_bytes()
    if (digest(cb61)!=SOURCE_CB61_SHA or digest(side_raw)!=S93_SIDECAR_SHA or
        digest(s142_raw)!=S142_SHA):
        raise AssertionError("Original trained S93/43M S142 physical source mismatch")
    raw=s87.open_envelope(cb61)
    cb=CB59(raw)
    if (cb.meta["original_unique_parameter_count"]!=BASE_36M or
        int(MODEL_SPEC[1])!=320 or int(MODEL_SPEC[3])!=12):
        raise AssertionError("Original 36.38M is not a 12x320 original CALIBRI core")
    s93=load_file(str(input.S93_sidecar))
    s142=load_file(str(input.S142_FP16))
    if not all(k in s142 and torch.equal(v,s142[k]) for k,v in s93.items()):
        raise AssertionError("43M S142 must preserve every S93 original GT/controller")
    if not (s142[IDS_KEY].dtype==torch.int32 and
        tuple(s142[IDS_KEY].tolist())==initial_active_indices(896) and
        s142[WEIGHTS_KEY].dtype==torch.float16 and
        tuple(s142[WEIGHTS_KEY].shape)==(12,320,48)):
        raise AssertionError("43M active 48/896 original S142 trained coefficient ABI mutated")
    if len(s142)!=36:
        raise AssertionError("Expected only original 36 physical S142 tensors")
    # Authenticate all numerical sidecar/bank/router/copy results with original
    # S99 source contract, rather than trusting names or fake zero weights.
    originals=SimpleNamespace(**{
        x:getattr(input,x) for x in
        ["S94_bank","S94_report","S95_router","S95_report",
         "S98_copy_net","S98_report","S93_CB61","S93_sidecar"]})
    p94,p95,p98,bank,router,copy_net=load_frozen_numeric_weights(originals)
    if not all(torch.isfinite(x).all() for x in s142.values() if x.is_floating_point()):
        raise AssertionError("Nonfinite original S142 learned weight")
    return cb,raw,s93,s142,bank,router,copy_net


class Stack:
    def __init__(self,cb,source,s142,*,dense=False):
        self.dense=dense
        if dense:
            self.model=s30.dense_from_paid(cb)
            self.sources={}
            # The original S91 frozen_controllers helper forces Apple MPS;
            # mirror actual S87 CPU loader with the SAME paid tensors.
            self.first,self.later=s78.controllers(s142)
            self.ptr=s82.GenericContextPointer()
            self.ptr.load_state_dict({
                k.removeprefix('generic_pointer.'):v for k,v in s142.items()
                if k.startswith('generic_pointer.')
            },strict=True)
            self.ptr.eval().requires_grad_(False)
            self.head=self.model.lm_head.weight.detach().cpu().contiguous()
            self.inner=IntegratedGreenTunnel(layers=(3,9),rank=8).eval()
            self.inner.load_state_dict({
               k.removeprefix("internal_tunnel."):v.float()
               for k,v in s142.items() if k.startswith("internal_tunnel.")
            },strict=True)
            install_verified_fastpath(self.inner)
            self.inner.install(self.model)
        else:
            self.model,self.sources,self.first,self.later,self.ptr,self.head,self.inner=(
                load_cpu_original(source,s142)[:7])
        self.model.eval().requires_grad_(False)
        self.bridge=PromptConditionedDualTunnel(
            self.inner,total_layers=12,donor_layers=(3,9)).eval().install(self.model)
        # S142 actually uses zero-initialized bridge coefficients. Do NOT
        # invent learned donor-gate tensors absent from real S142 physical.
        if "conditional_dual_bridge.coefficients" in s142:
            self.bridge.load_physical_gain({
                "conditional_dual_bridge.coefficients":
                s142["conditional_dual_bridge.coefficients"]})
        self.full=FullLayerStateTunnel(width=320,depth=12,rank=4).eval()
        load_compact_q2(self.full,{
           k:v for k,v in s142.items() if k.startswith(PREFIX)
        })
        self.full.install(self.model)
        self.progressive=(ProgressiveRouteCore(rank=896).eval()
                          .bind_original_route_bridge(self.bridge)
                          .install(self.model))
        self.progressive.load_active_fp16({
            IDS_KEY:s142[IDS_KEY],WEIGHTS_KEY:s142[WEIGHTS_KEY]})
        self.progressive.assert_inactive_zero()
        if self.progressive.architectural_new_parameter_count()!=EXTRA_43M:
            raise AssertionError("Not 43.261760M actual 12-layer architecture")
        self.submodules={
          "S87_active_GT3_9":tuple(self.inner.layers),
          "S111_prompt_donor":tuple(self.bridge.connected_layers),
          "S126_full_12":self.full.depth,
          "S132_core_additive_rank":self.progressive.rank,
          "S142_physically_stored_active_channels_per_layer":48,
        }
        self.head=self.head.detach().cpu().contiguous()

    def tokens(self,tok,question,feature,copy):
        with self.progressive.prefill_only(self.model),prefill_only_tunnel(
            self.model,self.full),self.bridge.prompt_scope("S93",feature):
            output,details=cached_full_cpu_with_neural_copy(
                self.model,self.sources,self.first,self.later,self.head,
                tok,question,self.ptr,copy)
        return output,details

    @torch.inference_mode()
    def layer_probe(self,tok,question,feature):
        prompt=tok.apply_chat_template(
            [{"role":"user","content":question}],
            tokenize=False,add_generation_prompt=True)
        ids=torch.tensor([tok.encode(prompt,add_special_tokens=False)],dtype=torch.long)
        clear(self.sources)
        with self.progressive.prefill_only(self.model),prefill_only_tunnel(
            self.model,self.full),self.bridge.prompt_scope("S93",feature):
            out=self.model(input_ids=ids,past_key_values=None,
                           use_cache=True,output_hidden_states=True)
        if (len(out.hidden_states)!=13 or
            self.full.last_forward_coverage!=tuple(range(12)) or
            self.progressive.last_prefill_coverage!=tuple(range(12))):
            raise AssertionError("Authentic S142 GT/43M full12 hook coverage failed")
        states=[v[0,-1].detach().float().cpu().clone()
                for v in out.hidden_states]
        pred=out.logits[0,-1].detach().float().cpu().clone()
        clear(self.sources)
        return states,pred

    def close(self):
        self.progressive.uninstall()
        self.full.uninstall()
        self.bridge.uninstall()
        self.inner.uninstall()


def selected_S93_existing_train(tok,router):
    """A priori choose first 8 train cases routed to active S93, not QA."""
    s30.TOK=tok
    rows=[]
    for family,pairs in sorted(s87.families(tok).items()):
        for pair in pairs:
            for side in ("left","right"):
                item=pair[side]
                q=item["question"]
                vec=torch.as_tensor(prompt_only_vector(tok,q),dtype=torch.float32)
                with torch.no_grad():
                    route=CHOICES[int(router(vec[None]).argmax())]
                if route!="S93":
                    continue
                expected=tok.encode(item["answer"],add_special_tokens=False)+[
                    tok.eos_token_id]
                rows.append({"TRAIN_id":f'{family}/{pair["pair_id"]}/{side}',
                             "question":q,"expected":expected,
                             "feature":vec,"route":route})
                if len(rows)==SELECTED_DEBUG:
                    return rows
    raise AssertionError("Not enough original trained S93 routing TRAIN cases")


def paired_outputs(dense,baseline,compressed,tok,rows,copy):
    reports=[]
    for idx,row in enumerate(rows):
        orig_h,orig_logits=dense.layer_probe(tok,row["question"],row["feature"])
        old_h,old_logits=baseline.layer_probe(tok,row["question"],row["feature"])
        new_h,new_logits=compressed.layer_probe(tok,row["question"],row["feature"])
        def layer_stats(a,b):
            return [{"layer":i-1,
                     "max_abs":float(torch.max(torch.abs(x-y))),
                     "relative_l2":float(torch.linalg.vector_norm(x-y)/
                                         torch.linalg.vector_norm(x).clamp_min(1e-12)),
                     "cosine":float(torch.nn.functional.cosine_similarity(
                                       x[None],y[None]).item())}
                     for i,(x,y) in enumerate(zip(a,b))]
        original_token,original_copy=dense.tokens(
            tok,row["question"],row["feature"],copy)
        old_token,old_copy=baseline.tokens(
            tok,row["question"],row["feature"],copy)
        new_token,new_copy=compressed.tokens(
            tok,row["question"],row["feature"],copy)
        pre={
            "TRAIN_id":row["TRAIN_id"],"question_sha":digest(row["question"].encode()),
            "routed_existing_S95_profile":row["route"],
            "original_expected_ids":row["expected"],
            "dense_FP32_mirror_ids":original_token,
            "real_S142_compressed_baseline_ids":old_token,
            "FP32_recompressed_candidate_ids":new_token,
            "dense_vs_original_S142_full_EOS_tokens_identical":original_token==old_token,
            "dense_vs_new_recompressed_all_token_equal":original_token==new_token,
            "already_working_43M_baseline_gold":old_token==row["expected"],
            "new_recompressed_43M_gold":new_token==row["expected"],
            "dense_greedy_text":tok.decode(original_token,skip_special_tokens=True),
            "baseline_greedy_text":tok.decode(old_token,skip_special_tokens=True),
            "recompressed_greedy_text":tok.decode(new_token,skip_special_tokens=True),
            "all_12_original_GT_hidden":layer_stats(orig_h,old_h),
            "all_12_new_recompressed_hidden":layer_stats(orig_h,new_h),
            "FP32_mirror_vs_original_GT_logits_maxabs":
               float(torch.max(torch.abs(orig_logits-old_logits))),
            "FP32_mirror_vs_RECODE_logits_maxabs":
               float(torch.max(torch.abs(orig_logits-new_logits))),
            "original_FP32_next_id":int(orig_logits.argmax()),
            "S142_original_next_id":int(old_logits.argmax()),
            "S7A_recompressed_next_id":int(new_logits.argmax()),
            "real_S98_original_policy_diagnostic":old_copy,
            "real_S98_recompressed_policy_diagnostic":new_copy,
        }
        log("S7A_actual_trained_43M_F32_vs_two_Calibri",
            row=idx+1,ident=row["TRAIN_id"],old_match=original_token==old_token,
            new_match=original_token==new_token,
            original_FP32=pre["dense_greedy_text"][:80],
            fresh_compressed=pre["recompressed_greedy_text"][:80],
            dense_new_final_cos=round(pre["all_12_new_recompressed_hidden"][-1]["cosine"],5))
        reports.append(pre)
    return reports


def run(args):
    torch.manual_seed(43071117)
    torch.set_num_threads(2)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    parent,source,s93,side,bank,router,copy=validate_parent(args)
    frozen={
      "experiment":"S7A authentic S142 original CALIBRI 43.26M matched FP32 reconstruction and then fresh physical compression",
      "original_43M_S142_safetensors_sha":S142_SHA,
      "source_S93_CB61_sha":SOURCE_CB61_SHA,
      "original_12layer_core":BASE_36M,
      "new_additive_original_896x12x320":EXTRA_43M,
      "original_total_43M_architecture":TOTAL_43M,
      "original_core_pretrained_independently_as_FP32":False,
      "matched_FP32_teacher_source":"EXACT DECODE OF THE ALREADY TRAINED ORIGINAL CALIBRI S93 compressed core PLUS original physically S142 trained 43M additive weights",
      "independently_pretrained_43M_FP32_model_loaded":False,
      "new_unseen_QA_or_EVAL_used":False,
      "all_sealed_135M_eval_unchanged":True,
      "original_S142_weights_and_GT_unchanged":True,
      "NRA_present":False,
      "skip_jessica_chat_testing":True,
      "stop_on_wrong_12layer_or_12hidden_or_S142_source_provenance":True
    }
    dump(out/"S7A_PRECOMMIT_BEFORE_MODEL_RECONSTRUCTION.json",frozen)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    rows=selected_S93_existing_train(tok,router)
    # Construct full dense *same-source* FP32 43.26M model and then really
    # requantize every dense matrix with the ordinary ORIGINAL K32/V8 codec.
    fp32_dense=s30.dense_from_paid(parent)
    count=sum(p.numel() for p in fp32_dense.parameters())
    if count!=BASE_36M or fp32_dense.config.num_hidden_layers!=12:
        raise AssertionError("Original CALIBRI cannot be reified to compatible 12-layer FP32")
    # Teacher includes 6,881,280 FP32 addon tensor cells in its live stack.
    new_raw,meta,original_weight_diagnostic=encode(fp32_dense.state_dict())
    if (meta["original_unique_parameter_count"]!=BASE_36M or
        len(new_raw)!=len(source) or new_raw==source):
        raise AssertionError("New independent FP32→CB59 conversion not physically distinct")
    new_archive=CB59(new_raw)
    if (new_archive.meta!=parent.meta or
        new_archive.handles["model.embed_tokens.weight"] is not
        new_archive.handles["lm_head.weight"]):
        raise AssertionError("New original 43M direct paid decoder/aliases incompatible")
    packed=new_CB61(new_raw)
    new_model=out/"S7A_FRESH_FP32_MIRROR_TO_CB61_43M.s61"
    new_model.write_bytes(packed)
    compressed=Stack(new_archive,new_raw,side)
    baseline=Stack(parent,source,side)
    dense=Stack(parent,source,side,dense=True)
    try:
        results=paired_outputs(dense,baseline,compressed,tok,rows,copy)
    finally:
        compressed.close()
        baseline.close()
        dense.close()
    prev_matches=sum(r["dense_vs_original_S142_full_EOS_tokens_identical"]
                     for r in results)
    novel_matches=sum(r["dense_vs_new_recompressed_all_token_equal"]
                      for r in results)
    max_logits=max(r["FP32_mirror_vs_original_GT_logits_maxabs"] for r in results)
    max_last_error=max(r["all_12_original_GT_hidden"][-1]["relative_l2"]
                         for r in results)
    # Original core decoded to FP32 should be numerically close to same FP32
    # centroids even after model hooks. Never silently call mismatches parity.
    if max_logits>0.01 or max_last_error>1e-3 or prev_matches!=len(results):
        raise AssertionError(
          f"Reifying original 43M CALIBRI to matched FP32 lost parity "
          f"(first={prev_matches}/{len(results)}, abslogit={max_logits}, hidden={max_last_error})")
    # One retained source plus 1 physical S142 checkpoint, one each of its
    # route bank/selector/copier, and a measured manifest. Dedupe old
    # S93 (already embedded in S142) and DO NOT charge original FP32 teacher.
    package_bytes={
      "fresh_recoded_CB61_bytes":len(packed),
      "original_43M_S142_all_neural_tensors_bytes":Path(args.S142_FP16).stat().st_size,
      "S94_three_profile_bank_bytes":Path(args.S94_bank).stat().st_size,
      "S95_neural_router_bytes":Path(args.S95_router).stat().st_size,
      "S98_second_token_copier_bytes":Path(args.S98_copy_net).stat().st_size,
      "config_runtime_compatibility_contract_bytes":0,
    }
    report={
      **frozen,
      "executed_real_12layer_trained_compressed_source":True,
      "original_existing_S142_full_456plus128_TRAIN_guard_from_prior":
           {"old456_full":362,"new128_full":17,
            "not_rerun_by_this_8prompt_experiment":True},
      "checked_known_S93_TRAIN_questions":len(results),
      "all_real_original_S142_vs_FP32_mirror_full_matches":prev_matches,
      "new_fresh_FP32_K32V8_compression_full_matches":novel_matches,
      "baseline_FP32_mirror_max_logit_diff":max_logits,
      "new_original_decoder_43M_parameters_excluding_GT_sidecars":TOTAL_43M,
      "new_true_full_uncompressed_FP32_decoder_and_additive_BYTES":
           4*TOTAL_43M,
      "all_12_original_and_requantized_hidden_states_and_real_generated_output":results,
      "codec_report":original_weight_diagnostic,
      "new_CB61_sha256":digest(packed),
      "new_CB59_underlying_SHA256":digest(new_raw),
      "new_model_file":new_model.name,
      "source_original_S93_CB61_bytes":Path(args.S93_CB61).stat().st_size,
      "original_S93_source_replace_not_duplicate":True,
      "original_FP32_teacher_not_shipped_inside_package":True,
      "S142_learnt_S87_GT3_9_full12_S126_router_copy_all_physically_reused":True,
      "source_only_K32V8_refit_not_new_GT_training":True,
      "new_FP32_independent_model_parity_proven":False,
      "new_135M_FP32_transfer_parity_proven":False,
      "no_native_C_90pct_speed_claim":True,
      "quality_gate_on_unseen_QA_not_opened":True,
      "ready_for_135M_application":False,
      "all_physically_paid_components":package_bytes,
      "candidate_accepted_for_promotion":False,
    }
    manifest=out/"S7A_CONVERTER_FULL_43M_REAL_OUTPUT_AND_PHYSICAL_REPORT.json"
    for iteration in range(20):
        dump(manifest,report)
        size=manifest.stat().st_size
        previous=package_bytes["config_runtime_compatibility_contract_bytes"]
        package_bytes["config_runtime_compatibility_contract_bytes"]=size
        bytes_total=sum(package_bytes.values())
        report["all_inclusive_recompressed_43M_model_bytes"]=bytes_total
        report["all_inclusive_recompressed_43M_bpp"]=8*bytes_total/TOTAL_43M
        report["physical_model_below_0p7"]=8*bytes_total/TOTAL_43M<=MAX_BPP
        report["conservative_rate_excluding_seed_generated_additive_43m_bpp"]=(
            8*bytes_total/BASE_36M)
        if size==previous:break
    else:
        raise AssertionError("S7A exact source physical manifest did not stabilize")
    dump(manifest,report)
    if (manifest.stat().st_size!=package_bytes["config_runtime_compatibility_contract_bytes"] or
        sum(package_bytes.values())!=report["all_inclusive_recompressed_43M_model_bytes"] or
        not report["physical_model_below_0p7"]):
        raise AssertionError("S7A complete 43M model physical >=0.7bpp")
    log("S7A_FINISHED_ACTUAL_43M_dense_FP32_to_CALIBRI",
        original_source_FP32_trained_independently=False,
        original_working_43M_roundtrip_full_EOS_matches=prev_matches,
        n=len(results),new_requantized_full_matches=novel_matches,
        max_original_dense_vs_CB59_logit_abs=max_logits,
        fresh_CB61_sha=digest(packed),
        full_physical_bpp=report["all_inclusive_recompressed_43M_bpp"],
        old_456_new128_retested=False,
        generalization_135M_proven=False)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for item in ("S93_CB61","S93_sidecar","S142_FP16","S94_bank",
                 "S94_report","S95_router","S95_report",
                 "S98_copy_net","S98_report","out"):
        ap.add_argument(item,type=Path)
    run(ap.parse_args())
