"""S18c7: ONE-TIME frozen original43M+GT native-C NEW32 EVAL32 readout.

Both paid control S18c1 rank8 2309 FP16 and grammar-aware S18c4 2597
FP16 were physically fixed BEFORE 32 EVAL texts/roles were authored.
First verify S18c6 immutable SHA and genuine original S194 616 J7 CPU
and TRUE Calibri43M GT post-layer8 32 raw EVAL (read-only).
Then execute original frozen CB59 43 packed C matrices + S11 unchanged.
ALL ordered wordpairs, both directions, including negatives; NO oracle
candidate filtering or gold in any neural model input.
Only AFTER both models' full logits are frozen/retained, construct
EVAL32 gold labels and calculate exact directed role P/R/F1, per-role
and per-construction. NO training, optimizer, calibration, search or
checkpoint selection. The first result permanently SPENDS EVAL32.
Not a representative natural Russian benchmark or certified LIVE.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from safetensors.torch import load_file
from direct_codebook_v59 import CB59
from research.nra3_s0.neural_area import NRA3Config
from research.nra3_s2.native_cpu import NRA3NativePackedCPU
from research.nra3_s11.paid_router import PaidThreeWayRouter
from research.nra3_s13.real43m_train72_readouts import S8B_43_CB59_SHA,S11_39_FP16_SHA
from research.nra3_s16.paid_directed_valence import (
    PaidDirectedValence,PARAMETERS as CONTROL_PARAMETERS,
    PHYSICAL_BYTES as CONTROL_BYTES, ROLES,
)
from research.nra3_s16.real43m_train72_directed_valence import (
    DirectedPairTrain,metrics,
)
from research.nra3_s18.s18c4_paid_rawgrammar_rank8 import (
    PaidRawGrammarDirectedValence,PARAMETERS as CANDIDATE_PARAMETERS,
    PHYSICAL_BYTES as CANDIDATE_BYTES,DIM,
)
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s18.s18c6_new32_one_time_eval_prereg import (
    build_new_eval32, seal_new_eval32_source,
    EVAL32_COMPLETE_CANONICAL_SHA256,
    PRE_FROZEN_CANDIDATE_2597_FP16_SHA,
    PRE_FROZEN_CONTROL_2309_FP16_SHA,
    CATEGORIES, EXPECTED_ROLES,EXPECTED_EDGES,
)
CAPTURE_RUN_SHA="16cb3964a0c4220afc82ff67756534a180623f0303c92aede6eff48e0416a5fe"
S18C6_FROZEN_CI=43919713933


def require(ok, reason):
    if not bool(ok):
        raise ValueError("S18c7 ONE TIME new32 original43M CPU native43 frozen readout: "+reason)


def physical_capture_rows(hidden_path:Path, report_path:Path):
    require(hidden_path.is_file() and report_path.is_file(),
            "full original real43M GT CPU eval32 feature/report absent")
    manifest=json.loads(report_path.read_text(encoding="utf-8"))
    hash_data=hashlib.sha256(hidden_path.read_bytes()).hexdigest()
    strict={
        "stage":"NRA3_S18C7_TRUE_ORIGINAL43M_GT_LAYER8_CPU_NEW32_ONE_TIME_EVAL_NO_GOLD",
        "S18c6_prereg_new32_complete_source_SHA":EVAL32_COMPLETE_CANONICAL_SHA256,
        "S18c6_first_source_only_CI":43919440242,
        "S18c6_new32_source_gold_links_never_to_original43M":104,
        "S18c7_old_consumed_EVALs_never_loaded":True,
        "S18c4_candidate_FP16_SHA_frozen_BEFORE_EVAL":PRE_FROZEN_CANDIDATE_2597_FP16_SHA,
        "S18c1_control_FP16_SHA_frozen_BEFORE_EVAL":PRE_FROZEN_CONTROL_2309_FP16_SHA,
        "S194_original_frozen_616_J7_parent_run":43345267876,
        "S194_original_checkpoint_SHA":"b1e58fde43bc9d4593f6db78f1c9b55d556255f4dcac7c96bce09302b68b34c2",
        "original_616_J7_cpu_generated_tokens_unchanged":True,
        "original_456_full_EOS":456,
        "original_new128_full_EOS":128,
        "original_32_train_source_correct":0,
        "actual_43M_CPU_12_layers":True,
        "actual_GreenTunnel_precedes_layer8_readport":True,
        "original_43M_GT_no_optimizer":True,
        "original_43M_GT_new32_original_CPU_prefills":32,
        "native_C_or_rank8_used_during_capture":False,
        "original_tokenizer_true_chat_offsets_all_32":True,
        "new_candidate_active_J7_EOS_tested":False,
        "new_candidate_full_system_bpp_le07_proven":False,
        "new_candidate_CPU_speed_ge90pct_CB59_proven":False,
        "original_43M_GT_and_readonly_616_J7_unchanged":True,
        "new_model_activated":False,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
        "LIVE_Jessica_Render_Supabase_APK_modified":False,
    }
    require(all(manifest.get(k)==v for k,v in strict.items()),
            "original real43M GT CPU, J7, new32 source, or no-activation contract drift")
    require(manifest["actual_fp16_feature_file"]==hidden_path.name
            and manifest["actual_fp16_feature_sha256"]==hash_data,
            "physical 32 EVAL original FP16 source SHA mismatch")
    t=load_file(str(hidden_path),device="cpu")
    require(set(t)=={"token_hidden320","input_ids","token_mask","token_lengths"},
            "EVAL labels or unrecognized data included inside original CPU activations")
    h=t["token_hidden320"];ids=t["input_ids"]
    mask=t["token_mask"];lengths=t["token_lengths"]
    require(h.ndim==3 and h.shape[0]==32 and h.shape[-1]==320
            and h.dtype==torch.float16 and ids.dtype==torch.int32
            and mask.dtype==torch.bool and lengths.shape==(32,)
            and ids.shape==mask.shape==h.shape[:2]
            and list(h.shape)==manifest["actual_hidden320_shape"]
            and lengths.tolist()==manifest["true_new32_eval_token_counts"]
            and bool(torch.isfinite(h).all()),
            "original physical hidden320/tokenizer IDs are wrong dimensions")
    rows=build_new_eval32()
    require(len(rows)==32 and len(manifest["wordpiece_alignment"])==32,
            "original new32 source or model wordpiece map count drift")
    saved_ids=[]
    for i,row in enumerate(rows):
        n=int(lengths[i])
        require(n>1 and n<=ids.shape[1]
                and bool(torch.equal(mask[i],torch.arange(ids.shape[1])<n))
                and bool((ids[i,n:]==-1).all()) and bool((h[i,n:]==0).all())
                and bool((ids[i,:n]>=0).all()),
                "original wordpiece IDs/padding corrupted")
        wp=manifest["wordpiece_alignment"][i]
        words=raw_surface_grammar(row.text)
        positions=wp.get("word_last_token_positions")
        pieces=wp.get("word_token_positions")
        require(wp.get("text_sha256")==row.text_sha256()
                and wp.get("token_count")==n
                and len(words)==len(positions)==len(pieces),
                "real original tokenizer raw EVAL word offset source identity drift")
        used=set();lasts=[]
        for last,ps in zip(positions,pieces):
            require(type(last) is int and isinstance(ps,list) and ps
                    and last==max(ps)
                    and all(type(k) is int and 0<=k<n and k not in used for k in ps),
                    "real original BPE piece reused across EVAL raw words")
            used.update(ps);lasts.append(last)
        require(lasts==sorted(set(lasts)),
                "actual original token word order not strictly increasing")
        saved_ids.append([int(x) for x in ids[i,:n].tolist()])
    actual_ids_sha=hashlib.sha256(json.dumps(
        saved_ids,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    require(actual_ids_sha==manifest["actual_token_ids_lengths_sha256"],
            "original Calibri new32 source true token ID digest mismatch")
    return rows,h,ids,lengths,manifest


def source_pair_cpu(rows,h,lengths,wp,native):
    """NEURAL INPUTS: original hidden320, actual BPE positions, RAW grammar.

    Do NOT read row.directed_gold, predicate_word_indices or its category
    inside original native CB59 C or the source 2 frozen FP16 readout heads.
    """
    pred=[];arg=[];pi=[];ai=[];now=[];ori=[];rowids=[];gpred=[];garg=[]
    relation_sha=[]
    with torch.inference_mode():
        for k,row in enumerate(rows):
            n=int(lengths[k])
            hidden=h[k,:n].float().contiguous()
            out=native(hidden.unsqueeze(0))
            require(out.relations.shape==(1,n,128)
                    and out.parallel_hypothesis_gates is not None
                    and out.parallel_hypothesis_gates.shape==(1,n,3)
                    and torch.isfinite(out.relations).all()
                    and native.output_gate==0 and native.candidate_accepted is False,
                    "not physically executing original 43 packed native-C and fixed S11")
            rel=out.relations[0].detach().float().contiguous()
            words=raw_surface_grammar(row.text)
            last=wp[k]["word_last_token_positions"]
            require(len(words)==len(last) and last==sorted(set(last))
                    and max(last)<n,"unaligned physical hidden to plain EVAL word")
            for i in range(len(words)):
                for j in range(i+1,len(words)):
                    for p,a in ((i,j),(j,i)):
                        t1,t2=int(last[p]),int(last[a])
                        pred.append(rel[t1]);arg.append(rel[t2])
                        pi.append(t1);ai.append(t2)
                        now.append(max(t1,t2))
                        ori.append(0 if p>a else 1)
                        rowids.append(k)
                        gpred.append(words[p].features)
                        garg.append(words[a].features)
            relation_sha.append(hashlib.sha256(
                rel.numpy().astype("<f4",copy=False).tobytes()).hexdigest())
    n=len(pred);aslong=lambda z:torch.tensor(z,dtype=torch.long)
    require(n>EXPECTED_EDGES*5
            and len(pred)==len(arg)==len(gpred)==len(garg)==len(rowids)
            and len(relation_sha)==32 and len(set(rowids))==32,
            "all original new32 EVAL wordpairs both directions not captured")
    pvec=torch.stack(pred).float().contiguous()
    avec=torch.stack(arg).float().contiguous()
    pi,ai,now,ori,rowids=(aslong(z) for z in (pi,ai,now,ori,rowids))
    gp=torch.tensor(gpred,dtype=torch.float32)
    ga=torch.tensor(garg,dtype=torch.float32)
    require(pvec.shape==avec.shape==(n,128)
            and gp.shape==ga.shape==(n,DIM)
            and bool(((gp==0)|(gp==1)).all())
            and bool(((ga==0)|(ga==1)).all())
            and bool((now==torch.maximum(pi,ai)).all()),
            "same genuine native source directed inputs not shared by both candidates")
    # Gold is deliberately DUMMY and never passed to model inference.
    dummy=DirectedPairTrain(pvec,avec,pi,ai,now,
                            torch.zeros(n,dtype=torch.long),
                            ori,rowids,32,0)
    return dummy,gp,ga,relation_sha


def labels_only_after_both_frozen_predictions(rows,dummy):
    """Build gold AFTER both actual frozen model forward passes succeeded."""
    labels=torch.zeros(dummy.total_pairs,dtype=torch.long)
    pairkeys=[{} for _ in range(32)]
    for k,row in enumerate(rows):
        # Derive word position for each RAW word by actual offset-derived
        # true BPE word:last-token index, present in current row pairs.
        srcmask=(dummy.source_index==k)
        where=srcmask.nonzero(as_tuple=False).view(-1).tolist()
        lasts=sorted(set(dummy.predicate_positions[where].tolist()+
                         dummy.argument_positions[where].tolist()))
        require(len(lasts)==len(raw_surface_grammar(row.text)),
                "one original token:word alignment per EVAL raw word required")
        coords={(int(dummy.predicate_positions[i]),int(dummy.argument_positions[i])):i
                for i in where}
        require(len(coords)==len(where),
                "duplicate directed candidate wordpair in old original source")
        expected=set()
        for z in row.directed_gold:
            key=(lasts[z.predicate_word-1],lasts[z.argument_word-1])
            require(key in coords, "one-time gold directed link absent from raw original pairs")
            ix=coords[key]
            labels[ix]=ROLES.index(z.role)
            require(ix not in expected,"one-time directed EVAL gold duplicate")
            expected.add(ix)
        pairkeys[k]=expected
    require(int((labels!=0).sum())==EXPECTED_EDGES==104
            and {ROLES[i]:int((labels==i).sum()) for i in range(1,5)}==EXPECTED_ROLES,
            "manually sealed 104 EVAL gold directed labels broken")
    return replace(dummy,target=labels,positive_links=104)


def subset_score(logits,x,rowids):
    mask=torch.zeros(x.rows,dtype=torch.bool)
    mask[list(rowids)]=True
    take=mask[x.source_index]
    sl=lambda val:val[take].detach().clone().contiguous()
    target=sl(x.target)
    sub=DirectedPairTrain(
       sl(x.predicate),sl(x.argument),sl(x.predicate_positions),
       sl(x.argument_positions),sl(x.current_positions),target,
       sl(x.orientation),sl(x.source_index),len(rowids),
       int((target!=0).sum()))
    require(sub.total_pairs==int(take.sum()),
            "source metrics subsets are not complete grouped directed wordpairs")
    return metrics(sl(logits),sub)


def main():
    p=argparse.ArgumentParser()
    for name in ("new32-hidden","new32-capture-report","s8-cb59",
                 "s11-router","s18c1-control","s18c4-candidate","out"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"do not overwrite original one-time new32 result")
    require(EVAL32_COMPLETE_CANONICAL_SHA256==CAPTURE_RUN_SHA
            and seal_new_eval32_source()["new_eval_source_complete_sha256"]==CAPTURE_RUN_SHA,
            "locked S18c6 EVAL32 SHA missing or drift")
    rows,h,ids,lengths,capture=physical_capture_rows(
         a.new32_hidden,a.new32_capture_report)
    data_control=a.s18c1_control.read_bytes()
    data_candidate=a.s18c4_candidate.read_bytes()
    require(CONTROL_PARAMETERS==2309 and CONTROL_BYTES==4665
            and CANDIDATE_PARAMETERS==2597 and CANDIDATE_BYTES==5286
            and hashlib.sha256(data_control).hexdigest()==PRE_FROZEN_CONTROL_2309_FP16_SHA
            and hashlib.sha256(data_candidate).hexdigest()==PRE_FROZEN_CANDIDATE_2597_FP16_SHA,
            "both original paid readout models were not frozen before EVAL32")
    control=PaidDirectedValence.from_physical(data_control).eval()
    candidate=PaidRawGrammarDirectedValence.from_physical(data_candidate).eval()
    require(control.physical_bytes()==data_control
            and candidate.physical_bytes()==data_candidate
            and all(not z.requires_grad for z in control.parameters())
            and all(not z.requires_grad for z in candidate.parameters())
            and control.output_gate==candidate.output_gate==0
            and control.candidate_accepted is False
            and candidate.candidate_accepted is False,
            "both frozen original candidates physically reloaded only for inference")
    packed=a.s8_cb59.read_bytes()
    router=a.s11_router.read_bytes()
    require(len(packed)==653798 and hashlib.sha256(packed).hexdigest()==S8B_43_CB59_SHA
            and len(CB59(packed).handles)==43
            and CB59(packed).meta["original_unique_parameter_count"]==6_106_154
            and len(router)==123
            and hashlib.sha256(router).hexdigest()==S11_39_FP16_SHA
            and PaidThreeWayRouter.from_physical(router).physical_bytes()==router,
            "original genuine 43 packed native-C or 39FP16 S11 SHA changed")
    a.out.mkdir(parents=True,exist_ok=False)
    library=a.out/"s18c7_original43_CB59_nativeC.so"
    subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",
                   "research/calibri/direct_kernel_v59.c","-o",str(library)],check=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    native=NRA3NativePackedCPU(
        packed,NRA3Config(parallel_hypotheses=3,
            parallel_selector_version=3),library,paid_router_physical=router).eval()
    require(native.source_sha256==S8B_43_CB59_SHA
            and native.paid_router.physical_bytes()==router
            and not native.native_cpu_weights_dense_resident
            and native.output_gate==0 and native.candidate_accepted is False
            and all(not v.requires_grad for v in native.parameters()),
            "original native43 CB59+S11 weights not frozen")
    # Neural input reader is raw-only. No gold inspection before inference.
    dummy, gp,ga, relation_sha=source_pair_cpu(rows,h,lengths,capture["wordpiece_alignment"],native)
    with torch.inference_mode():
        control_logits=control.score_pairs(
            dummy.predicate,dummy.argument,dummy.predicate_positions,
            dummy.argument_positions,dummy.current_positions).detach().clone()
        candidate_logits=candidate.score_pairs(
            dummy.predicate,dummy.argument,dummy.predicate_positions,
            dummy.argument_positions,dummy.current_positions,gp,ga).detach().clone()
    require(control_logits.shape==candidate_logits.shape==(dummy.total_pairs,5)
            and bool(torch.isfinite(control_logits).all())
            and bool(torch.isfinite(candidate_logits).all())
            and control.physical_bytes()==data_control
            and candidate.physical_bytes()==data_candidate,
            "both original FP16 frozen predictions not fully complete before EVAL labels")
    # Exactly now one-time sealed EVAL gold is accessed for FIRST score.
    scored=labels_only_after_both_frozen_predictions(rows,dummy)
    full_control=metrics(control_logits,scored)
    full_candidate=metrics(candidate_logits,scored)
    categories=[]
    for kind in CATEGORIES:
        indices=tuple(k for k,row in enumerate(rows) if row.category==kind)
        require(len(indices)==4,"sealed one-time EVAL32 category group missing")
        old=subset_score(control_logits,scored,indices)
        new=subset_score(candidate_logits,scored,indices)
        categories.append({
            "category":kind,"EVAL_sentences":len(indices),
            "control_2309FP16":old["all_two_direction_wordpair_exact_role"],
            "candidate_2597FP16":new["all_two_direction_wordpair_exact_role"],
            "gold_positive":old["all_two_direction_wordpair_exact_role"]["gold_links"],
        })
    require(len(categories)==8
            and sum(cat["EVAL_sentences"] for cat in categories)==32
            and sum(cat["gold_positive"] for cat in categories)==104,
            "all eight NEW EVAL32 diagnostic construction groups not scored exactly once")
    fullold=full_control["all_two_direction_wordpair_exact_role"]
    fullnew=full_candidate["all_two_direction_wordpair_exact_role"]
    report={
        "stage":"NRA3_S18C7_TRUE43M_GT_NATIVE43_S11_NEW32_ONE_TIME_EVAL_FINAL_SCORED_SPENT",
        "S18c6_source_only_prereg_SHA":CAPTURE_RUN_SHA,
        "S18c6_first_prereg_CI":43919440242,
        "S18c6_SHA_locked_CI":S18C6_FROZEN_CI,
        "S18c4_frozen_physical_2597FP16_SHA":PRE_FROZEN_CANDIDATE_2597_FP16_SHA,
        "S18c1_frozen_physical_2309FP16_SHA":PRE_FROZEN_CONTROL_2309_FP16_SHA,
        "original_S194_616_J7_unchanged_PARENT_only":True,
        "frozen_original_calibri43M_GT_true_CPU_prefills":32,
        "real_frozen_original43M_GT_hidden_FP16_SHA":capture["actual_fp16_feature_sha256"],
        "original_native_CB59_43_C_SHA":S8B_43_CB59_SHA,
        "original_S11_39_FP16_SHA":S11_39_FP16_SHA,
        "native_C_relation32_SHA":relation_sha,
        "actual_EVAL32_complete_raw_bidirectional_wordpairs":scored.total_pairs,
        "actual_one_time_directed_semantic_gold_links":scored.positive_links,
        "EVAL32_complete_original_text_rows":len(rows),
        "control_2309_fp16":{"full_exact_directional_roles":full_control},
        "candidate_2597_fp16":{"full_exact_directional_roles":full_candidate},
        "comparison":{
            "control_P":fullold["precision"],"control_R":fullold["recall"],
            "control_F1":fullold["f1"],
            "candidate_P":fullnew["precision"],"candidate_R":fullnew["recall"],
            "candidate_F1":fullnew["f1"],
            "candidate_delta_F1":fullnew["f1"]-fullold["f1"],
            "candidate_better_than_control_on_SPENT_EVAL32":fullnew["f1"]>fullold["f1"],
        },
        "per_construct_category":categories,
        "original_native43_frozen_without_gold_inference":True,
        "no_oracle_gold_upos_deprel_lemma_predicate_filtering":True,
        "both_frozen_FP16_predictions_completed_BEFORE_gold_labels_built":True,
        "all_bidirectional_negative_pairs_included":True,
        "both_models_fixed_BEFORE_EVAL_source_authored":True,
        "this_one_time_NEW32_EVAL_NOW_SPENT_NEVER_TUNE_OR_REUSE":True,
        "this_handwritten_NEW32_not_representative_natural_Russian_corpus":True,
        "no_optimizer_training_or_checkpoint_selection":True,
        "no_prior_S12c_S15c_S17c_EVAL128_or_reserved_S3CI_loaded":True,
        "strict_online_left_to_right_parse_unproven_due_S95_full_prompt":True,
        "new_candidate_616_J7_EOS_NOT_tested":True,
        "inclusive_bpp_le07_NOT_certified":True,
        "CPU_speed_ge90pct_CB59_NOT_certified":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
        "LIVE_Jessica_Render_Supabase_APK_modified":False,
    }
    out=a.out/"NRA3_S18C7_ONE_TIME_NEW32_SPENT_EVAL_2309_VS_2597_NO_SELECTION.json"
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",
                   encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items()
          if k not in ("control_2309_fp16","candidate_2597_fp16",
                       "native_C_relation32_SHA")},ensure_ascii=False,
          sort_keys=True),flush=True)


if __name__=="__main__":
    main()
