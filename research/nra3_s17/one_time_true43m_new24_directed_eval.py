"""NRA2 S17c: ONE-TIME frozen S16b physical trained directed-link NEW24 eval.

Freeze order: original43M+GT, 43 original C-packed CB59 matrices,
S11 physically learned 39FP16 3way selector, and S16b 2,309FP16
directional valence checkpoint were independently frozen BEFORE S17
new 24 Russian EVAL sentences were written. S17 FULL annotations SHA
was then preregistered BEFORE any 43M forward. S17b captures genuine
layer8 source states under original protected J7 baseline without labels.

IMPORTANT: the original frozen S95 prompt_only_vector hashes ALL tokens
of the complete sentence, and S111 dual-GT prompt_scope applies that
FULL-PROMPT feature to layer hooks before the NRA3 read port. Thus
the per-word native readout DOES NOT necessarily provide a strictly
online-left-to-right representation, even though the rank8 scorer
itself indexes only the individual two original wordpiece states.
This is a valid whole-sentence directed role parsing diagnostic,
NOT proof of online causality/streaming comprehension.

This script MUST NOT run unless real S17b FP16 source SHA is pinned.
There is no training, calibration, checkpoint selection, and no data
from previously consumed S15c EVAL24/S12c EVAL16 or EVAL128.

Scoring is one-time: send ALL directed ordered wordpairs, including
punctuation and non-verbs, through physically frozen S16b *before*
opening ANY directed gold. Gold is used only afterward for precision,
recall, F1; wrong class counts FP+FN, NONE class contributes no TP.
Event negation/future, pronoun coreference and answer EOS are NOT
output heads and their accuracy is NOT evaluated by this diagnostic.
"""
from __future__ import annotations
from dataclasses import dataclass
import argparse
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
from research.nra3_s16.paid_directed_valence import (
    PaidDirectedValence,PARAMETERS,PHYSICAL_BYTES,ROLES
)
from research.nra3_s17.new_ru24_directed_eval_prereg import (
    S17_EVAL24_COMPLETE_SHA,build_new_eval24,preregister_new_eval24,
    gold_links,SPLIT
)

S17B_FROZEN_ORIGINAL43M_CPU_HIDDEN_SHA="22af4b8420bc0091fd53dad10be1c5e8c0700a09f08e006b043419567d8e44d4" # immutable genuine CPU SUCCESS run 43900663076 before any S17c score
S17_GOLD_SHA="fca54736aa555d9c5b3665d1bb9a1fc1979ad7abdeb083af015d90dc05ed1a32"
S16B_FROZEN_MODEL_SHA="3fe4bbce718e53e4e812e94367ce25834f873cbed646479c59b1e3949d2ecf38"
S16B_TRAIN_SOURCE_SHA="9843c35ee9aa75b8f5c9c963e441589dd2f0292e41ec7933971d5aa3dcd365d6"
S8B_FROZEN_CB59_SHA="d710aed710a44719b6003434a2dd83150861928ce167364cb92329beabf7be97"
S11_FROZEN_39_FP16_SHA="05ce14e20c9f4e7e09f85dbd9bd46f0dc0a700d72f1a96ffa921b2e822f235c3"
S17B_REAL_ORIGINAL_SOURCE_CPU_RUN=43900663076 # immutable genuine CPU SUCCESS SHA verified before NEW EVAL scoring


def require(b,reason):
    if not bool(b):
        raise ValueError("NRA3 S17c NEW24 one-time post-SHA frozen validation: "+reason)


def sha_file(p:Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


@dataclass(frozen=True)
class OriginalEvalWordpiece:
    row:object
    hidden320:torch.Tensor
    true_wordpiece_last_tokens:tuple[int,...]
    original_token_ids:tuple[int,...]


def original_cpu_new24_source(hidden_path:Path,report_path:Path):
    require(bool(S17B_FROZEN_ORIGINAL43M_CPU_HIDDEN_SHA)
            and len(S17B_FROZEN_ORIGINAL43M_CPU_HIDDEN_SHA)==64
            and S17B_REAL_ORIGINAL_SOURCE_CPU_RUN>0,
            "DO NOT run before pinning FIRST original S17b true43M hidden FP16 and run SHA")
    require(hidden_path.is_file() and report_path.is_file()
            and sha_file(hidden_path)==S17B_FROZEN_ORIGINAL43M_CPU_HIDDEN_SHA,
            "actual original43M post-GT layer8 physical SHA drift")
    r=json.loads(report_path.read_text(encoding="utf-8"))
    expected={
      "stage":"NRA3_S17B_TRUE_ORIGINAL_CALIBRI43M_GT_LAYER8_NEW_DIRECTED_RU24_READONLY",
      "S17_NEW_RUSSIAN24_DIRECTED_COMPLETE_UD_GOLD_SHA256":S17_GOLD_SHA,
      "S17_SOURCE_ONLY_FIRST_PREREG_RUN":43898185861,
      "S17_COMPLETE_GOLD_SHA_PREREG_BEFORE_ORIGINAL43M":True,
      "S17_S16B_FROZEN_2309FP16_SOURCE_SHA_BEFORE_NEW_EVAL":S16B_FROZEN_MODEL_SHA,
      "S17_FROZEN_2309FP16_ADAPTER_NOT_LOADED":True,
      "S17_new_EVAL24_model_selection_performed":False,
      "S17_new_EVAL24_directed_gold_NOT_passed_to_original43M":True,
      "S17_new_EVAL24_scores_measured":False,
      "all_616_native_CPU_generated_token_digests_unchanged":True,
      "original_456_full_EOS":456,
      "original_new128_full_EOS":128,
      "original_new32_full_EOS":0,
      "actual_43M_CPU_12_transformer_layers":True,
      "observed_layer_index_zero_based":7,
      "GT_residuals_before_readport_in_same_active_S93_scope":True,
      "true_real_Calibri_token_alignment_all_24":True,
      "NRA3_output_gate":0.,
      "NRA3_candidate_accepted":False,
      "real_source_eval24_count":24,
      "actual_genuine_43M_CPU_prefills":24,
      "no_training_or_optimizer_for_original_43M_and_GT":True,
      "all_gold_labels_excluded_from_frozen_decoder":True,
      "actual_fp16_feature_sha256":S17B_FROZEN_ORIGINAL43M_CPU_HIDDEN_SHA,
      "old_S15c_EVAL24_and_S12c_EVAL16_consumed_sources_not_loaded":True,
      "five_consumed_independent_EVAL128_not_loaded":True,
      "sixth_reserved_S3CI_EVAL128_not_loaded":True,
      "no_Render_Supabase_APK_LIVE_or_Jessica_chat":True,
    }
    require(all(r.get(k)==v for k,v in expected.items()),
            "S17b true original43M/J7 and source-only gold precommit report mismatch")
    require(S17_EVAL24_COMPLETE_SHA==S17_GOLD_SHA and
            preregister_new_eval24()[
                "S17_new_Russian_EVAL_complete_UD_and_directed_link_SHA256"
            ]==S17_GOLD_SHA,
            "24 NEW independently preregistered directed labels changed")
    dat=load_file(str(hidden_path),device="cpu")
    require(set(dat)=={"token_hidden320","input_ids","token_mask","token_lengths"},
            "gold/EVAL annotations must NOT be embedded in original43M hidden input")
    h=dat["token_hidden320"]; ids=dat["input_ids"]
    valid=dat["token_mask"]; lengths=dat["token_lengths"]
    require(h.ndim==3 and h.shape[0]==24 and h.shape[2]==320
            and h.dtype==torch.float16
            and bool(torch.isfinite(h).all())
            and ids.dtype==torch.int32 and valid.dtype==torch.bool
            and ids.shape==valid.shape==h.shape[:2]
            and lengths.shape==(24,)
            and list(h.shape)==r["actual_hidden320_shape"]
            and lengths.tolist()==r["true_eval24_token_counts"],
            "original real43M 24 token lengths/FP16 hidden ABI changed")
    rows=build_new_eval24()
    maps=r["wordpiece_alignment"]
    require(len(rows)==len(maps)==24,"24 source/wordpiece rows required")
    src=[];real_id_order=[]
    for k,(row,m) in enumerate(zip(rows,maps)):
        t=int(lengths[k])
        require(1<t<=h.shape[1] and
                torch.equal(valid[k],torch.arange(h.shape[1])<t)
                and bool((ids[k,:t]>=0).all()) and
                bool((ids[k,t:]==-1).all()) and
                bool((h[k,t:]==0).all()) and
                m.get("text_sha256")==row.text_sha256()
                and m.get("token_count")==t,
                "real original Calibri wordpiece/source row identity or padding drift")
        all_ids=tuple(int(x) for x in ids[k,:t].tolist())
        real_id_order.append(list(all_ids))
        ps=m["word_last_token_positions"]
        pieces=m["word_token_positions"]
        require(len(ps)==len(pieces)==len(row.words)
                and all(type(j) is int and 0<=j<t for j in ps)
                and ps==sorted(set(ps))
                and all(type(v) is list and v and max(v)==j
                        for v,j in zip(pieces,ps))
                and sum(len(v) for v in pieces)==len({a for v in pieces for a in v}),
                "actual original Calibri EVAL24 token offsets overlap or leak future")
        src.append(OriginalEvalWordpiece(
            row,h[k,:t].float().contiguous(),tuple(ps),all_ids))
    canonical=json.dumps(real_id_order,sort_keys=True,
                         separators=(",",":")).encode()
    require(hashlib.sha256(canonical).hexdigest()==
            r["actual_token_ids_lengths_sha256"],
            "actual original 43M input ID SHA diverged")
    return tuple(src),r


def frozen_physical_sources(original:bytes,router39:bytes,valence:bytes,train_report:dict):
    require(len(original)==653798 and
            hashlib.sha256(original).hexdigest()==S8B_FROZEN_CB59_SHA and
            len(CB59(original).handles)==43 and
            CB59(original).meta["original_unique_parameter_count"]==6_106_154,
            "original CB59 43 C-packed 6.1M matrices changed")
    require(len(router39)==123 and
            hashlib.sha256(router39).hexdigest()==S11_FROZEN_39_FP16_SHA and
            PaidThreeWayRouter.from_physical(router39).physical_bytes()==router39,
            "original physically paid 39FP16 S11 selector changed")
    require(len(valence)==PHYSICAL_BYTES==4665 and
            hashlib.sha256(valence).hexdigest()==S16B_FROZEN_MODEL_SHA,
            "trained S16b source changed after NEW24 EVAL prereg")
    require(train_report.get("stage")==
              "NRA3_S16B_TRUE_ORIGINAL43M_GT_RU72_216_DIRECTED_LINKS_NATIVE_C_TRAIN_ONLY"
            and train_report.get("S16b_NEW_2309_paid_FP16_SHA")==S16B_FROZEN_MODEL_SHA
            and train_report.get("S16b_original_43_CB59_native_C_matrices_unchanged") is True
            and train_report.get("S16b_selected_because_of_TRAIN_only") is True
            and train_report.get("S16b_selected_TRAIN_only_FP16_step")==200
            and train_report.get("S16b_positive_directed_role_gold")==216
            and train_report.get("S16b_separate_future_NEW_independent_Russian_EVAL_performed") is False
            and train_report.get("S16b_previously_consumed_S15c_EVAL24_not_loaded") is True
            and train_report.get("S16b_previously_consumed_S12c_EVAL16_not_loaded") is True,
            "S16b froze only original 72 TRAIN before NEW24 EVAL")
    model=PaidDirectedValence.from_physical(valence).eval()
    require(sum(x.numel() for x in model.parameters())==PARAMETERS==2309
            and all(not x.requires_grad for x in model.parameters())
            and model.output_gate==0 and model.candidate_accepted is False,
            "2,309 paid FP16 source must be frozen with zero gate")
    return model


def calc_exact_directed_role_metrics(trials:list[tuple[object,tuple[tuple[int,int],...],torch.Tensor]]):
    """After ALL frozen native/scorer forwards, open gold once for reporting."""
    require(len(trials)==24,"no cherry-picking NEW EVAL24 sentences")
    accum=[];groups={};count_by_role={name:{"TP":0,"FP":0,"FN":0} for name in ROLES[1:]}
    trials_summary=[]
    for row,candidates,logits in trials:
        require(row.split==SPLIT
                and len(candidates)==len(row.words)*(len(row.words)-1)
                and logits.shape==(len(candidates),len(ROLES))
                and bool(torch.isfinite(logits).all()),
                "all ordered original Calibri words INCLUDING negatives mandatory")
        gold={(p,a):ROLES.index(role) for p,a,role in gold_links(row)}
        pred=logits.argmax(-1).tolist()
        expected=torch.tensor([gold.get((p,a),0) for p,a in candidates],dtype=torch.long)
        correct=int(sum(p==g and g!=0 for p,g in zip(pred,expected.tolist())))
        predicted=int(sum(p!=0 for p in pred))
        n_gold=int((expected!=0).sum())
        fp=int(sum(p!=g and p!=0 for p,g in zip(pred,expected.tolist())))
        fn=int(sum(p!=g and g!=0 for p,g in zip(pred,expected.tolist())))
        require(len(gold)==n_gold and
                correct+fn==n_gold and correct+fp==predicted,
                "exact directed role FP/FN arithmetic drift")
        group=groups.setdefault(row.event_modality,{"sentences":0,"TP":0,"FP":0,"FN":0,"gold":0})
        group["sentences"]+=1
        group["TP"]+=correct;group["FP"]+=fp;group["FN"]+=fn;group["gold"]+=n_gold
        for role in ROLES[1:]:
            rid=ROLES.index(role)
            r=count_by_role[role]
            for guess,truth in zip(pred,expected.tolist()):
                r["TP"]+=int(guess==truth==rid)
                r["FP"]+=int(guess==rid and truth!=rid)
                r["FN"]+=int(truth==rid and guess!=rid)
        orientation={"predicate_now":{"TP":0,"FP":0,"FN":0,"gold":0},
                     "argument_now":{"TP":0,"FP":0,"FN":0,"gold":0}}
        for (p,a),guess,truth in zip(candidates,pred,expected.tolist()):
            typ="predicate_now" if p>a else "argument_now"
            z=orientation[typ]
            z["TP"]+=int(guess==truth and truth!=0)
            z["FP"]+=int(guess!=truth and guess!=0)
            z["FN"]+=int(guess!=truth and truth!=0)
            z["gold"]+=int(truth!=0)
            accum.append((guess,truth,typ))
        trials_summary.append({
            "construction":row.construction,"new_eval_text_sha256":row.text_sha256(),
            "source_words":len(row.words),
            "all_ordered_directional_pairs":len(candidates),
            "gold_positive_directed_edges":n_gold,
            "correct_exact_positive_edges":correct,
            "predicted_positive_edges":predicted,
            "FP":fp,"FN":fn,
            "event_modality_only_gold_metadata_not_predicted":row.event_modality,
            "gold_pronoun_coreference_never_predicted":any(w.upos=="PRON" for w in row.words)
        })
    tp=sum(int(a==b and b!=0) for a,b,_ in accum)
    fp=sum(int(a!=b and a!=0) for a,b,_ in accum)
    fn=sum(int(a!=b and b!=0) for a,b,_ in accum)
    gold_count=tp+fn
    pred_count=tp+fp
    require(gold_count==74 and len(trials_summary)==24 and
            len(accum)>24*25 and tp<=74,
            "all 74 NEW direct gold arcs and unordered+ordered negatives required")
    def score_counts(tp,fp,fn):
        prec=tp/(tp+fp) if tp+fp else 0.
        recall=tp/(tp+fn) if tp+fn else 0.
        f1=2*prec*recall/(prec+recall) if prec+recall else 0.
        return {"tp":tp,"fp":fp,"fn":fn,"precision":prec,"recall":recall,"f1":f1}
    by_orientation={name:score_counts(
        sum(a==b and b!=0 and o==name for a,b,o in accum),
        sum(a!=b and a!=0 and o==name for a,b,o in accum),
        sum(a!=b and b!=0 and o==name for a,b,o in accum)
      ) for name in ("predicate_now","argument_now")}
    report=score_counts(tp,fp,fn)
    return {
        "S17c_independent_NEW24_gold_true_directed_edges":gold_count,
        "S17c_new24_directed_ordered_allpairs_including_all_negatives":len(accum),
        "S17c_all24_frozen_directed_role_micro":report,
        "S17c_per_role":{r:score_counts(v["TP"],v["FP"],v["FN"])
                         for r,v in count_by_role.items()},
        "S17c_by_surface_order":by_orientation,
        "S17c_by_gold_event_modality_ROLE_ONLY_not_event_truth":{
             k:{**v,**score_counts(v["TP"],v["FP"],v["FN"])}
             for k,v in groups.items()},
        "S17c_per_sentence_readonly_diagnostic":trials_summary,
    }


def main():
    ap=argparse.ArgumentParser()
    for field in ("s17b-hidden","s17b-report","s8-cb59",
                  "s11-router","s16b-valence","s16b-report","out"):
        ap.add_argument("--"+field,type=Path,required=True)
    a=ap.parse_args()
    require(not a.out.exists(),
            "S17c NEW24 ONE-TIME result cannot overwrite or create a second score")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    samples,original_report=original_cpu_new24_source(
        a.s17b_hidden,a.s17b_report)
    cb59=a.s8_cb59.read_bytes();router=a.s11_router.read_bytes()
    physical=a.s16b_valence.read_bytes()
    train_report=json.loads(a.s16b_report.read_text(encoding="utf-8"))
    valence=frozen_physical_sources(cb59,router,physical,train_report)
    a.out.mkdir(parents=True,exist_ok=False)
    so=a.out/"original43_nra3_CB59_native_C_s17c.so"
    subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",
                    "research/calibri/direct_kernel_v59.c","-o",str(so)],check=True)
    native=NRA3NativePackedCPU(cb59,
          NRA3Config(parallel_hypotheses=3,parallel_selector_version=3),
          so,paid_router_physical=router).eval()
    require(native.source_sha256==S8B_FROZEN_CB59_SHA
            and native.paid_router.physical_bytes()==router
            and not native.native_cpu_weights_dense_resident
            and native.output_gate==0 and native.candidate_accepted is False,
            "full original 43 native packed CB59+S11 C CPU shadow mandatory")

    # The complete 24 EVAL source gold is NOT passed to the candidate
    # scorer; enumerate every ordered pair in both directions up front.
    trials=[]
    with torch.inference_mode():
        for sample in samples:
            t=sample.hidden320.shape[0]
            original=native(sample.hidden320.unsqueeze(0))
            require(original.relations.shape==(1,t,128) and
                    original.parallel_hypothesis_gates is not None
                    and original.parallel_hypothesis_gates.shape==(1,t,3)
                    and bool(torch.isfinite(original.relations).all()),
                    "true original 43 native-C matrices and frozen S11 did not run")
            allpos=sample.true_wordpiece_last_tokens
            n=len(allpos)
            # ALL pairs, including punctuation & nonverbs, no gold labels.
            candidates=tuple((p,q) for p in range(1,n+1)
                             for q in range(1,n+1) if p!=q)
            ppos=torch.tensor([allpos[p-1] for p,q in candidates],dtype=torch.long)
            apos=torch.tensor([allpos[q-1] for p,q in candidates],dtype=torch.long)
            current=torch.maximum(ppos,apos)
            scores=valence.score_pairs(
                original.relations[0,ppos,:].detach(),
                original.relations[0,apos,:].detach(),
                ppos,apos,current)
            require(scores.shape==(len(candidates),len(ROLES))
                    and bool(torch.isfinite(scores).all()),
                    "frozen model must score ALL directed candidate pairs")
            trials.append((sample.row,candidates,scores.detach().clone()))
    require(len(trials)==24,"must finish ALL 24 original43M native CPU forwards")
    # NO MORE MODEL CALLS OR OPTIMIZER AFTER THIS POINT.
    score=calc_exact_directed_role_metrics(trials)
    require(sha_file(a.s8_cb59)==S8B_FROZEN_CB59_SHA and
            sha_file(a.s11_router)==S11_FROZEN_39_FP16_SHA and
            sha_file(a.s16b_valence)==S16B_FROZEN_MODEL_SHA and
            valence.physical_bytes()==physical,
            "source SHA drift before/after frozen one-time EVAL")
    bpp=8*(len(cb59)+len(router)+len(physical))/(6_106_154+39+PARAMETERS)
    report={
      "stage":"NRA3_S17C_ONE_TIME_TRUE43M_NEW24_FROZEN_S16B_DIRECTED_ROLE_EVAL",
      "S17b_original_real43M_GT_CPU_hidden_SHA":S17B_FROZEN_ORIGINAL43M_CPU_HIDDEN_SHA,
      "S17b_original_parent_616_J7_baseline_run":S17B_REAL_ORIGINAL_SOURCE_CPU_RUN,
      "S17_NEW24_COMPLETE_UD_DIRECTED_GOLD_PREREG_SHA":S17_GOLD_SHA,
      "S17_SOURCE_ONLY_CI_BEFORE_FIRST_MODEL_EVAL":43898826121,
      "S16b_candidate_SHA_FROZEN_BEFORE_S17_new_EVAL24":S16B_FROZEN_MODEL_SHA,
      "S16b_frozen_FP16_physical_bytes":len(physical),
      "S16b_frozen_FP16_parameter_count":PARAMETERS,
      "original_S8B_43_CB59_C_matrices_unchanged_SHA":S8B_FROZEN_CB59_SHA,
      "original_S11_39FP16_router_unchanged_SHA":S11_FROZEN_39_FP16_SHA,
      "S17c_true_original43M_GT_NATIVE_C_all24_scored_once":True,
      "S17c_no_gold_POS_verb_or_direct_link_candidate_filter":True,
      "S17c_native_NRA_pair_scorer_only_reads_own_wordpiece_positions":True,
      "S17c_upstream_frozen_GT_S95_gate_conditions_on_FULL_PROMPT":True,
      "S17c_STRICT_ONLINE_CAUSALITY_end_to_end_certified":False,
      "S17c_actual_scoring_all_24_after_model_SHA_freeze":True,
      "S17c_no_EVAL24_training_or_checkpoint_selection":True,
      "S17c_EVAL24_optimizer_steps":0,
      "S17c_new_EVAL24_now_CONSUMED_no_future_retesting_or_tuning":True,
      "S17c_original_parent_616_J7_baseline_unchanged":True,
      "S17c_active_S16b_candidate_616_J7_or_fullEOS_tested":False,
      "S17c_negation_future_event_truth_prediction_measured":False,
      "S17c_pronoun_antecedent_coreference_prediction_measured":False,
      "S17c_RussianQA_or_dialogue_generation_quality_measured":False,
      "S17c_small_handannotated24_not_general_population_eval":True,
      "S17c_native_full_system_43M_GT_NRA_speed_at_least90pct_certified":False,
      "S17c_full_inclusive_original43M_GT_NRA_bpp_at_most07_certified":False,
      "S17c_standalone_raw_NRA_plus_S11_plus_S16b_bpp":bpp,
      "S17c_no_prior_consumed_S15c_EVAL24_or_S12c_EVAL16_used":True,
      "S17c_no_prior_five_EVAL128_or_reserved_sixth_S3CI_used":True,
      "S17c_no_LIVE_Jessica_chat_Render_Supabase_APK":True,
      "research_acceptance":False,
      "candidate_accepted":False,
      "activation_allowed":False,
      "output_gate":0,
      **score
    }
    result=a.out/"NRA3_S17C_ONE_TIME_NEW24_DIRECTED_ROLE_RESEARCH_NO_PROMOTION.json"
    result.write_text(json.dumps(report,sort_keys=True,ensure_ascii=False,
        indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in
          ("S17c_per_sentence_readonly_diagnostic",)},
          ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
