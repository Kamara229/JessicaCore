"""S119q2h: DIAGNOSE native language-logit ranking, NO model training.

The previous q2g research (already scored/EXPOSED, thus spent) yielded
0/12 full Russian answers for frozen S142 Core12+GT 43M, trained q2f
contextual NRA3/NPA1 and matched local. Its reported English "Yes"/
"yellow" fragments must be investigated at the *native full-vocabulary
head*, not hand-waved as semantic-area failures.

Re-open EXACT q2g 12 source cases and verify its actual run result SHA;
physically run original S142 43M+GreenTunnel full 49,152 LM head once
per source and each candidate (36 original decoder forwards). Record
full-vocab top16 BPE ranks/Unicode piece hints, native expected first
answer token rank, top1-vs-correct margin and language script counts.
Verify first physical greedy native token agrees with the recorded q2g
own-token generation for each SAME exact SHA-bound source prompt.

The GOLD short answer is used exclusively in POST-forward logit audit
and never enters the model prefix. No optimizer/hyperparameter tuning.
A single decoded BPE fragment is NOT a robust word/language classifier.
The q2g questions and labels have ALREADY been inspected; THIS IS
NOT an independent blind QA or new generalization result. No model
deployment, no J7/EVAL128/S118 protected data, no full bpp/CPU claims.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path

import torch

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    load_actual_original,sha,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    old_original_param_digest,
)
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import (
    first_original_pass,prompt_features,
)
from research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation import (
    strict_chat_source_ids,
)
from research.calibri_nra3.s119q1_real43m_onepass_causal_early import (
    onepass_full_native_logits,
)
from research.calibri_nra3.s119q2b_real43m_onepass_native_train import (
    verify_q2a_artifact,
)
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    REPORT as Q2D_REPORT, CONTEXT_CKPT as Q2D_CTX,
    LOCAL_CKPT as Q2D_LOCAL, VOCAB_SIZE,
)
from research.calibri_nra3.s119q2d1_real_original_sameprobe_baseline import (
    load_q2d_proof,
)
from research.calibri_nra3.s119q2e_causal_early_view_diagnostic import (
    load_q2d_trained_child,
)
from research.calibri_nra3.s119q2f_learned_causal_early_projection import (
    REPORT as Q2F_REPORT, CONTEXT_CKPT as Q2F_CTX,
    LOCAL_CKPT as Q2F_LOCAL, STAGE as Q2F_STAGE,
)
from research.calibri_nra3.s119q2f1_physical_projector_onoff import (
    load_physical_q2f_child,
)
from research.calibri_nra3.s119q2g_fresh_own_token_source import (
    rows as q2g_rows,validate as verify_q2g_source,
)
from research.calibri_nra3.s119q2g_physical_own_token_generation import (
    STAGE as Q2G_STAGE, REPORT as Q2G_REPORT,
    MODELS, SAMPLE_COUNT,
)

STAGE="CALIBRI_S119Q2H_REAL43M_NATIVE_RUSSIAN_LANGUAGE_LOGIT_FORENSICS"
REPORT="S119Q2H_REAL43M_NATIVE_LOGIT_LANGUAGE_FIRST_TOKEN_12X3.json"
SEED=119139
TOP_K=16
EXPECTED_PHYSICAL_CALLS=SAMPLE_COUNT*len(MODELS)


def native_piece_bucket(piece):
    """Script hint on ONE decoded BPE token, not language diagnosis."""
    check(type(piece) is str,"native Unicode decoded piece only")
    cyrillic=any("\u0400"<=c<="\u052f" for c in piece)
    latin=any("a"<=c.lower()<="z" for c in piece)
    if cyrillic and latin:
        return "mixed_script_piece"
    if cyrillic:
        return "cyrillic_piece"
    if latin:
        return "latin_piece"
    return "other_punctuation_special_or_incomplete_byte_piece"


def first_answer_native(tok,question,answer):
    """The ACTUAL native BPE after original chat header; no target leakage."""
    check(type(question) is str and type(answer) is str
          and question and answer,"source literal strings only")
    native_prompt=strict_chat_source_ids(tok,question)
    full_chat=tok.apply_chat_template(
        [{"role":"user","content":question}],
        tokenize=False,add_generation_prompt=True)
    native_full=list(tok.encode(full_chat+answer,add_special_tokens=False))
    # Honest boundary guarantee: if tokenizer merges assistant header with
    # answer, the naive answer-only first BPE is NOT the correct token.
    check(len(native_prompt)>=2 and native_full[:len(native_prompt)]==
          native_prompt and len(native_full)>len(native_prompt),
          "exact full-chat+answer native BPE must preserve causal prompt boundary")
    suffix=native_full[len(native_prompt):]
    check(0<len(suffix)<70 and
          all(type(x) is int and 0<=x<VOCAB_SIZE for x in suffix)
          and tok.decode(native_full,skip_special_tokens=False).endswith(answer),
          "exact native full-answer target ids, no proxy/special token error")
    return native_prompt,suffix


def physical_rank(tok,logits,target_native_id,expected_first_id):
    """Extract numerical native head evidence without modifying logits."""
    check(logits.shape==(1,VOCAB_SIZE) and
          bool(torch.isfinite(logits).all()) and
          0<=target_native_id<VOCAB_SIZE and
          0<=expected_first_id<VOCAB_SIZE and TOP_K==16,
          "real original 49,152-class raw head ranking only")
    one=logits[0].detach().float().cpu()
    top_ids=torch.topk(one,k=TOP_K,largest=True,sorted=True).indices.tolist()
    first=int(torch.argmax(one).item())
    check(first==top_ids[0] and first==expected_first_id,
          "original per-source first native BPE MUST reproduce q2g actual own generation")
    topone=float(one[first].item())
    gold=float(one[target_native_id].item())
    rank=int((one>one[target_native_id]).sum().item())+1
    pieces=[tok.decode([int(x)],skip_special_tokens=False)
            for x in top_ids]
    buckets=[native_piece_bucket(x) for x in pieces]
    probs=torch.log_softmax(one,dim=-1)
    return {
        "top1_native_id":first,
        "top1_native_text_piece":pieces[0],
        "top1_native_piece_bucket":buckets[0],
        "top1_native_logit":topone,
        "gold_FIRST_native_target_token_id":target_native_id,
        "gold_FIRST_native_piece":tok.decode(
            [target_native_id],skip_special_tokens=False),
        "gold_FIRST_native_rank_49152":rank,
        "gold_FIRST_native_logit":gold,
        "top1_minus_gold_first_logit_margin":topone-gold,
        "gold_first_BPE_probability_under_full_native_head":
            float(torch.exp(probs[target_native_id]).item()),
        "top16_native_ids":top_ids,
        "top16_decoded_BPE_pieces":pieces,
        "top16_script_hints":dict(Counter(buckets)),
        "top16_native_contains_gold_FIRST_token":target_native_id in top_ids,
        "script_hint_is_NOT_full_word_or_language_accuracy":True,
        "original_full_native_vocabulary":VOCAB_SIZE,
    }


def summarize_kind(entries):
    check(len(entries)==SAMPLE_COUNT and
          Counter(x["kind"] for x in entries)=={
              "roles_word_order":2,"negation":2,
              "transitive_comparison":2,"event_time":2,
              "arithmetic":2,"paraphrase":2},
          "exact earlier spent 12 source prompts, not a new independent QA test")
    a=[x["rank_evidence"] for x in entries]
    return {
        "n_source_prompts":len(entries),
        "median_gold_first_native_rank":
            float(torch.tensor(sorted(
                x["gold_FIRST_native_rank_49152"] for x in a),
                dtype=torch.float32).median().item()),
        "mean_gold_first_native_rank":
            sum(x["gold_FIRST_native_rank_49152"] for x in a)/len(a),
        "mean_top1_minus_gold_first_margin":
            sum(x["top1_minus_gold_first_logit_margin"] for x in a)/len(a),
        "gold_native_FIRST_in_top16":
            sum(x["top16_native_contains_gold_FIRST_token"] for x in a),
        "top1_piece_hints":dict(Counter(
            x["top1_native_piece_bucket"] for x in a)),
        "by_source_domain":{
            kind:{
                "mean_gold_first_rank":
                    sum(x["rank_evidence"]["gold_FIRST_native_rank_49152"]
                        for x in entries if x["kind"]==kind)/2,
                "top16_gold_first_count":
                    sum(x["rank_evidence"][
                        "top16_native_contains_gold_FIRST_token"]
                        for x in entries if x["kind"]==kind),
            } for kind in sorted({x["kind"] for x in entries})
        },
        "not_blind_and_no_new_training_or_hyperparameter_selection":True,
    }


def run(s93,side,s142,q2a_dir,q2d_dir,q2f_dir,q2g_dir,out):
    check(not out.exists(),"fresh isolated old-source forensic report, never overwrite")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    source=q2g_rows()
    provenance=verify_q2g_source(source)
    check(provenance["source_example_count"]==SAMPLE_COUNT,
          "only frozen exact already-scored 12 q2g Russian questions")
    gpath=q2g_dir/Q2G_REPORT
    previous=json.loads(gpath.read_text(encoding="utf8"))
    check(previous["stage"]==Q2G_STAGE and
          previous["new_source_provenance"]["twelve_source_sha256"]==
              provenance["twelve_source_sha256"] and
          previous["actual_original_decoder_forward_calls"]==208 and
          previous["original_S93_sha256"]==sha(s93) and
          previous["original_S142_43M_GT_sha256"]==sha(s142) and
          previous["native_greedy_OWN_emitted_BPE_only_no_gold_teacher_forcing"]
          and previous["candidate_accepted"] is False,
          "q2g SOURCE already physically scored, not a new blind holdout")
    old_rows,oldmeta=verify_q2a_artifact(q2a_dir)
    old_q2d=load_q2d_proof(
        q2d_dir/Q2D_REPORT,q2d_dir/Q2D_CTX,q2d_dir/Q2D_LOCAL,s93,s142)
    q2f=json.loads((q2f_dir/Q2F_REPORT).read_text(encoding="utf8"))
    check(q2f["stage"]==Q2F_STAGE and
          q2f["original_S93_SHA256"]==sha(s93) and
          q2f["original_S142_GT_SHA256"]==sha(s142) and
          q2f["original_q2a_synthetic_source_SHA256"]==
              oldmeta["synthetic_rows_SHA256"] and
          q2f["parent_q2d_training_report_SHA256"]==
              sha(q2d_dir/Q2D_REPORT) and
          previous["q2f_physical_checkpoint_report_sha256"]==
              sha(q2f_dir/Q2F_REPORT) and
          q2f["candidate_accepted"] is False and q2f["output_gate"]==0,
          "exact original q2f learned child and previous own generation SHA")
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==VOCAB_SIZE and tok.eos_token_id is not None,
          "unchanged original 49,152 native BPE tokenizer and real EOS")
    parents=[]
    for _ in (0,1):
        parent,_=load_q2d_trained_child(
            q2d_dir/Q2D_CTX,
            old_q2d["new_research_checkpoint_contextual"])
        parents.append(parent)
    contextual=load_physical_q2f_child(
        q2f_dir/Q2F_CTX,q2f["projector_contextual_child_checkpoint"],
        parents[0])
    local=load_physical_q2f_child(
        q2f_dir/Q2F_LOCAL,q2f["projector_local_matched_checkpoint"],
        parents[1])
    original,sources,bridge,full,expansion=load_actual_original(
        s93,side,s142)
    original.model.add_module("nra3_s119m_onpath",contextual)
    original.eval()
    sha_original=old_original_param_digest(original)
    snap_ctx={k:v.detach().clone().cpu() for k,v in contextual.state_dict().items()}
    snap_local={k:v.detach().clone().cpu() for k,v in local.state_dict().items()}
    check(all(not p.requires_grad for n,p in original.named_parameters()
              if "nra3_s119m_onpath" not in n),
          "the real pretrained original Core12/GreenTunnel completely frozen")
    traces={key:[] for key in MODELS}
    physical=0
    for row,old in zip(source,previous["source_only_12_new_author_questions_answers"]):
        question=row["question"]
        check(old["source_id"]==row["source_id"] and
              old["expected_short_answer"]==row["canonical_short_answer"]
              and old["question_sha256"]==
                  hashlib.sha256(question.encode("utf8")).hexdigest(),
              "must audit SAME previously exposed own-generation source ordering")
        prompt,answer_native=first_answer_native(
            tok,question,row["canonical_short_answer"])
        features=prompt_features(tok,question)
        check(features.shape==(512,) and len(prompt)<160,
              "full original native input/question only, no answer tokens")
        first_gold=int(answer_native[0])
        for variant in MODELS:
            child=(contextual if variant=="trained_contextual"
                   else local if variant=="trained_token_local" else None)
            if child is None:
                logits,original_layers=first_original_pass(
                    original,sources,bridge,full,expansion,prompt,features)
                del original_layers
            else:
                check(child.output_gate==0 and
                      not child.candidate_accepted and
                      child.projection_enabled,
                      "trained q2f child inference only, activation forbidden")
                out_logits,witness=onepass_full_native_logits(
                    original,sources,bridge,full,expansion,child,
                    prompt,features,
                    contextual=(variant=="trained_contextual"))
                check(witness["candidate_forward_calls"]==1,
                      "ONE original decoder forward per first token ranking")
                logits=out_logits.detach().float().cpu()
                del out_logits
            old_token=int(old["results"][variant]["generated_native_token_ids"][0])
            metrics=physical_rank(tok,logits,first_gold,old_token)
            traces[variant].append({
                "source_id":row["source_id"],
                "kind":row["source_kind"],
                "question_sha256":old["question_sha256"],
                "first_gold_native_target_token_id":first_gold,
                "gold_short_answer_native_BPE_tokens_count":len(answer_native),
                "gold_token_NEVER_in_model_input":True,
                "rank_evidence":metrics,
            })
            physical+=1
            del logits
    check(physical==EXPECTED_PHYSICAL_CALLS and
          old_original_param_digest(original)==sha_original and
          all(torch.equal(v,contextual.state_dict()[k].detach().cpu())
              for k,v in snap_ctx.items()) and
          all(torch.equal(v,local.state_dict()[k].detach().cpu())
              for k,v in snap_local.items()),
          "36 physical original-native forwards and all trained/original weights immutable")
    summary={variant:summarize_kind(traces[variant]) for variant in MODELS}
    result={
        "stage":STAGE,
        "S93_SHA256":sha(s93),
        "S142_43M_GT_SHA256":sha(s142),
        "q2f_checkpoint_report_SHA256":sha(q2f_dir/Q2F_REPORT),
        "previous_q2g_real_own_token_report_SHA256":sha(gpath),
        "already_scored_source_sha256":provenance["twelve_source_sha256"],
        "actual_original_full_native_decoder_calls":physical,
        "same_question_and_same_true_first_gold_native_BPE_for_all_models":True,
        "first_candidate_greedy_token_reproduced_previous_real_q2g_generation":True,
        "all_native_head_classes":VOCAB_SIZE,
        "topK":TOP_K,
        "summary":summary,
        "per_model_native_first_token_language_ranking":traces,
        "source_PREVIOUSLY_SCORED_not_independent_and_not_fresh":True,
        "no_optimizer_no_label_leakage_no_hyperparameter_selection":True,
        "script_classification_is_only_fragment_hint":True,
        "not_full_native_answer_generation_this_stage_previous_q2g_was":True,
        "original_Core12_GT_and_old_53751_FP16_UNCHANGED":True,
        "both_trained_q2f_contextual_local_fp16_UNCHANGED":True,
        "no_S118_B_GOLD_spent_EVAL128_616J7_or_Jessica_chat":True,
        "no_matched_KV_CPU_throughput_or_allin_packed_bpp_cert":True,
        "no_LIVE_Render_Supabase_main_APK_deploy":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"NATIVE_HEAD_LANGUAGE_ALIGNMENT_FORENSICS_ONLY_NO_ACCEPTANCE",
    }
    out.mkdir(parents=True)
    (out/REPORT).write_text(json.dumps(
        result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2H_REAL43M_NATIVE_LANGUAGE_LOGIT_FORENSICS "+
          json.dumps({
              "real_original_decoder_forwards":physical,
              "prior_new_author_source_SHA256":provenance["twelve_source_sha256"],
              "metrics":{
                  v:{
                      "mean_gold_native_rank":
                          summary[v]["mean_gold_first_native_rank"],
                      "gold_token_in_top16":
                          summary[v]["gold_native_FIRST_in_top16"],
                      "mean_top1_vs_gold_margin":
                          summary[v]["mean_top1_minus_gold_first_margin"],
                      "top1_script_piece_hints":
                          summary[v]["top1_piece_hints"],
                  } for v in MODELS
              },
              "no_learning_or_promotion":True,
          },sort_keys=True,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    for name in ("s93","side","s142","q2a_dir","q2d_dir","q2f_dir",
                 "q2g_dir","out"):
        parser.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=parser.parse_args()
    run(a.s93,a.side,a.s142,a.q2a_dir,a.q2d_dir,a.q2f_dir,
        a.q2g_dir,a.out)


if __name__=="__main__":
    main()
