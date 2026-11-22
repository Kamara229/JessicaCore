"""S119q2i5 physical native UTF8/BPE SEQUENCE first-divergence diagnosis.

No new model training. Read-only full 49,152 native LM-head analysis on
6 *ALREADY SPENT* original q2i1 DEV prompts, one from each six task
domains. Compare actual q2i1 zero-GT12 learned state, actual q2i3 joint
recurrent GT12, and q2i3 joint no-carry GT12 learned state.

Two distinct causal paths on SAME original S142 43M+GT native model:
 GOLD-PREFIX (teacher-forced DIAGNOSTIC ONLY): at every target position
   feed only gold PREVIOUS native BPEs, evaluate true next BPE rank,
   cross entropy, EOS and top1 with real original decoder forward.
 SELF-PREFIX: at every generation position feed ONLY previously
   greedy emitted OWN native tokens; reproduce q2i3 actual full
   12-token self-generation exactly, track first divergence from true
   native sequence, decoding of partial-byte token 155, and prefix
   UTF8 validity. Gold answer tokens NEVER enter self-generation
   decoder inputs. They are used only to score after forward.

Full native 49152 logits at EVERY STEP. Number of original forward
calls is dynamic but counted exactly (up to 3x6x(48+12)).
One-time source-only explanatory audit; previously scored non-blind DEV,
never use these diagnostics as future training or tuning labels.
Full 616 J7+EOS, bpp and efficient KV CPU NOT assessed. All existing
original pretrained and research FP16 checkpoint tensors unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re

import torch
import torch.nn.functional as F

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    load_actual_original,sha,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    old_original_param_digest,
)
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    CONTEXT_CKPT as Q2D_CTX,LOCAL_CKPT as Q2D_LOCAL,
    REPORT as Q2D_REPORT,VOCAB_SIZE,
)
from research.calibri_nra3.s119q2d1_real_original_sameprobe_baseline import (
    load_q2d_proof,
)
from research.calibri_nra3.s119q2e_causal_early_view_diagnostic import (
    load_q2d_trained_child,
)
from research.calibri_nra3.s119q2f_learned_causal_early_projection import (
    CONTEXT_CKPT as Q2F_CTX, REPORT as Q2F_REPORT,
)
from research.calibri_nra3.s119q2f1_physical_projector_onoff import (
    load_physical_q2f_child,
)
from research.calibri_nra3.s119q2i1_authored_ru_train_source import (
    rows as authored_rows,provenance as author_provenance,KINDS,
)
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import (
    STAGE as Q2I1_STAGE,CONTEXT_CKPT as Q2I1_CTX,REPORT as Q2I1_REPORT,
    verified_native_answer,real_joint_native,
)
from research.calibri_nra3.s119q2i2_gt12_recurrence_connectivity import (
    load_q2i1_physical_context,native_full_gt_gate_contract,set_gt_route,
)
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import (
    STAGE as Q2I3_STAGE, REPORT as Q2I3_REPORT,
    RECURRENT_CKPT,LOCAL_CKPT,RECURRENT_GATES,LOCAL_GATES,HEAD_GAIN,
    scoped_reduced_native_head_gain,
)
from research.calibri_nra3.s119q2i4_native_byte_head_component_forensics import (
    check_trained_joint, verified_gt12_gates,
)

STAGE="CALIBRI_S119Q2I5_ORIGINAL43M_NATIVE_GOLD_VS_OWN_UTF8_BPE_FIRST_DIVERGENCE"
REPORT="S119Q2I5_REAL43M_NATIVE_BYTE_UTF8_GOLD_VS_SELF_FIRST_DIVERGENCE.json"
SEED=119144
SAMPLE_PER_DOMAIN=1
SOURCE_COUNT=6
MODELS=("GT_ZERO_BASELINE","GT_RECURRENT","GT_NOCARRY_LOCAL")
MAX_SELF_TOKENS=12
MAX_GOLD_TARGET_TOKENS=48
BYTE_PIECE_PATTERN=re.compile(r"<0x([0-9A-Fa-f]{2})>")
EXPECTED_TRAINED_PARENT_FORWARDS=988


def native_bytelevel_full_utf8_status(tokenizer,token_ids):
    """Reconstruct actual GPT2/ByteLevel raw token bytes, across ALL BPEs.

    Original tokenizer raw pieces are e.g. 'Ð' (actual byte D0), 'Ŀ'
    (actual byte 9D), 'Ñģ' (actual D1 81), NOT necessarily '<0xD0>'.
    A single 'Ð' is an EXPECTED incomplete Russian first-byte prefix.
    Using per-piece tok.decode() alone produces transient replacement
    glyphs, not proof of an incorrectly emitted token.
    """
    bs=list(range(ord("!"),ord("~")+1))+list(
        range(ord("¡"),ord("¬")+1))+list(
        range(ord("®"),ord("ÿ")+1))
    cs=list(bs);n=0
    for b in range(256):
        if b not in bs:
            bs.append(b);cs.append(256+n);n+=1
    reverse={chr(c):b for b,c in zip(bs,cs)}
    stream=bytearray()
    unknown=[]
    rawpieces=[str(tokenizer.convert_ids_to_tokens(int(i)))
               for i in token_ids]
    special_ids=set(getattr(tokenizer,"all_special_ids",[]))
    for native_id,rawpiece in zip(token_ids,rawpieces):
        if int(native_id) in special_ids or (
            rawpiece.startswith("<|") and rawpiece.endswith("|>")):
            continue
        m=BYTE_PIECE_PATTERN.fullmatch(rawpiece)
        if m is not None:
            stream.append(int(m.group(1),16))
            continue
        for char in rawpiece:
            if char not in reverse:
                unknown.append(char)
                continue
            stream.append(reverse[char])
    status="VALID_COMPLETE_UTF8"
    if unknown:
        status="UNRECOGNIZED_TOKENIZER_RAW_GLYPHS"
    else:
        try:
            bytes(stream).decode("utf-8",errors="strict")
        except UnicodeDecodeError as exc:
            status=("INCOMPLETE_UTF8_NEEDS_CONTINUATION"
                    if exc.reason=="unexpected end of data"
                    else "INVALID_UTF8_SEQUENCE")
    return {
        "original_native_raw_BPE_pieces":rawpieces,
        "original_reconstructed_byte_prefix_hex":stream.hex().upper(),
        "entire_original_native_BPE_prefix_UTF8_status":status,
        "unrecognized_raw_glyphs":unknown,
        "raw_single_byte_D0_is_legitimate_Russian_prefix":True,
        "decoder_replacement_character_alone_not_quality_proof":True,
    }


def utf8_byte_tail_status(tokenizer,token_ids):
    """Evidence about contiguous trailing raw byte tokens; NOT all text."""
    pieces=[str(tokenizer.convert_ids_to_tokens(int(i)))
            for i in token_ids]
    tail=[]
    for piece in reversed(pieces):
        m=BYTE_PIECE_PATTERN.fullmatch(piece)
        if m is None:
            break
        tail.append(int(m.group(1),16))
    tail.reverse()
    decoded=tokenizer.decode(token_ids,skip_special_tokens=True)
    if not tail:
        state="NO_TRAILING_NATIVE_BYTE_TOKENS"
    else:
        try:
            bytes(tail).decode("utf-8",errors="strict")
            state="TRAILING_BYTES_VALID_UTF8"
        except UnicodeDecodeError as exc:
            state=("TRAILING_BYTES_INCOMPLETE_UTF8"
                   if exc.reason=="unexpected end of data"
                   else "TRAILING_BYTES_INVALID_UTF8")
    return {
        "native_token_ids":[int(i) for i in token_ids],
        "native_raw_token_strings":pieces,
        "contiguous_trailing_raw_byte_hex":[f"{x:02X}" for x in tail],
        "trailing_native_byte_state":state,
        "decoded_text_if_rendered_NOW":decoded,
        "replacement_UFFFD_visible_if_rendered_NOW":"\uFFFD" in decoded,
        "incomplete_byte_token_MAY_be_correct_native_prefix":True,
        "not_language_or_semantic_classification":True,
        "full_native_GPT2_BYTELEVEL_utf8_status":
            native_bytelevel_full_utf8_status(tokenizer,token_ids),
    }


def first_divergence(gold,own):
    check(type(gold) is list and type(own) is list
          and len(gold)>=2 and len(own)>=1
          and all(type(x) is int and 0<=x<VOCAB_SIZE
                  for x in gold+own),
          "must compare actual gold full native BPE+EOS vs actual own IDs")
    longest=min(len(gold),len(own))
    offset=next((i for i in range(longest)
                 if gold[i]!=own[i]),None)
    if offset is None and len(own)<len(gold):
        offset=len(own)
    elif offset is None and len(gold)<len(own):
        offset=len(gold)
    return {
        "first_wrong_or_missing_native_BPE_position_1based":
            (offset+1 if offset is not None else None),
        "matching_native_BPE_prefix_length":
            (offset if offset is not None else longest),
        "first_native_BPE_correct":
            (gold[0]==own[0]),
        "first_native_correct_but_wrong_OR_missing_second":
            (len(gold)>1 and gold[0]==own[0] and
             (len(own)<2 or gold[1]!=own[1])),
        "full_native_ID_sequence_EXACT":gold==own,
        "gold_NATIVE_sequence_contains_EOS":True,
    }


def decode_step_metrics(tok,logits,gold_id,step_index):
    check(logits.shape==(1,VOCAB_SIZE)
          and bool(torch.isfinite(logits).all())
          and 0<=gold_id<VOCAB_SIZE,
          "actual S142 49,152 native classes and real target BPE")
    v=logits[0].detach().float().cpu()
    predicted=int(torch.argmax(v))
    return {
        "native_position_1based":step_index+1,
        "correct_native_BPE_target_id":gold_id,
        "predicted_native_BPE_top1_id":predicted,
        "native_target_is_top1":predicted==gold_id,
        "native_target_rank_full_49152":
            int(torch.sum(v>v[gold_id]))+1,
        "native_target_CE":float(F.cross_entropy(
            v.view(1,-1),torch.tensor([gold_id])).item()),
        "native_top1_minus_gold_logit":float(v[predicted]-v[gold_id]),
        "gold_native_raw_token":str(tok.convert_ids_to_tokens(gold_id)),
        "predicted_native_raw_token":str(tok.convert_ids_to_tokens(predicted)),
        "native_vocab":VOCAB_SIZE,
    }


def measure_sequence(tok,model,sources,bridge,full,expansion,
                     child,example,previous_own,*,variant):
    check(example["split"]=="DEV" and
          variant in MODELS and
          HEAD_GAIN==.25 and
          len(example["answer_native_with_eos"])<=MAX_GOLD_TARGET_TOKENS and
          len(previous_own)>=1 and len(previous_own)<=MAX_SELF_TOKENS,
          "only q2i1 previously SCORED DEV and real prior q2i3 own native trace")
    gold=example["answer_native_with_eos"]
    check(int(gold[-1])==tok.eos_token_id
          and tok.eos_token_id not in gold[:-1],
          "target contains genuine terminal native EOS once")
    physical_forwards=0
    gold_steps=[]
    own_steps=[]
    own_tokens=[]
    # GOLD-PREFIX: each target position is evaluated on ONLY its prior
    # teacher-forced gold tokens. Not a model-generated semantic score.
    for i,target in enumerate(gold):
        prefix=example["prefix"]+gold[:i]
        with scoped_reduced_native_head_gain(child.language_head) as proof:
            logits=real_joint_native(
                model,sources,bridge,full,expansion,child,
                prefix,example["prompt_features"],contextual=True)
        check(proof[0]==1
              and full.last_forward_coverage==tuple(range(12))
              and not full._active and full._carry is None,
              "one original native full12 Core/GT/head forward per gold BPE")
        gold_steps.append(decode_step_metrics(tok,logits,int(target),i))
        physical_forwards+=1
        del logits
    # OWN-PREFIX: entirely self-produced autoregressive BPE tokens,
    # never gold or expected answers in the model prefix.
    for i in range(MAX_SELF_TOKENS):
        own_prefix=example["prefix"]+own_tokens
        check(own_prefix[len(example["prefix"]):]==own_tokens
              and len(own_prefix)<160,
              "native own-path cannot include a future or correct gold token")
        with scoped_reduced_native_head_gain(child.language_head) as proof:
            logits=real_joint_native(
                model,sources,bridge,full,expansion,child,
                own_prefix,example["prompt_features"],contextual=True)
        check(proof[0]==1 and
              full.last_forward_coverage==tuple(range(12))
              and not full._active and full._carry is None,
              "one original full native Core12 GT12 forward per self BPE")
        native=int(torch.argmax(logits.detach(),dim=-1))
        check(native==int(previous_own[i]),
              "physical SELF continuation must EXACTLY reproduce previously "
              "recorded q2i3 real 43M self-generated native token at each step")
        own_tokens.append(native)
        state=utf8_byte_tail_status(tok,own_tokens)
        own_steps.append({
            "self_native_step_1based":i+1,
            "actually_generated_native_id":native,
            "target_NATIVE_id_at_SAME_POSITION_if_it_exists":
                int(gold[i]) if i<len(gold) else None,
            "self_prefix_render_state":state,
            "top1_native_margin":float(
                (torch.topk(logits.detach().float().cpu()[0],2).values[0]-
                 torch.topk(logits.detach().float().cpu()[0],2).values[1])),
            "real_original_native_decoder_forwards":1,
        })
        physical_forwards+=1
        del logits
        if native==tok.eos_token_id:
            break
    check(own_tokens==previous_own and
          physical_forwards==len(gold_steps)+len(own_steps),
          "all previously scored native own tokens physically replayed "
          "and every diagnostic teacher-forced BPE physically measured")
    divergence=first_divergence(gold,own_tokens)
    return {
        "source_id":example["id"],"kind":example["kind"],
        "variant":variant,
        "true_gold_full_native_BPE_and_terminal_EOS":gold,
        "actual_self_generated_native_BPE_and_EOS_if_any":own_tokens,
        "actual_self_generated_own_native_TEXT":
            tok.decode(own_tokens,skip_special_tokens=True),
        "teacher_forced_gold_position_details":gold_steps,
        "free_running_self_position_details":own_steps,
        "gold_complete_native_sequence_byte_status":
            utf8_byte_tail_status(tok,gold),
        "actual_own_final_UTF8_native_byte_status":
            utf8_byte_tail_status(tok,own_tokens),
        "first_divergence":divergence,
        "number_gold_teacher_forced_full_original_decoder_forwards":
            len(gold_steps),
        "number_SELF_prefix_native_full_original_decoder_forwards":
            len(own_steps),
        "total_original_S142_Core12_full_native_decoder_forwards":
            physical_forwards,
        "not_new_independent_source_or_free_answer_quality_claim":True,
        "no_target_current_or_future_BPE_in_SELF_prefix":True,
    }


def summarize_sequences(records):
    check(len(records)==SOURCE_COUNT
          and Counter(x["kind"] for x in records)=={k:1 for k in KINDS},
          "exactly ONE old q2i1 DEV per each six original task domains")
    all_golds=[x for rec in records
               for x in rec["teacher_forced_gold_position_details"]]
    firsts=[r["first_divergence"][
                "first_wrong_or_missing_native_BPE_position_1based"]
            for r in records]
    return {
        "scored_old_DEV_source_examples":len(records),
        "correct_native_first_BPE":sum(
            r["first_divergence"]["first_native_BPE_correct"]
            for r in records),
        "correct_native_first_BPE_BUT_failed_second":sum(
            r["first_divergence"][
                "first_native_correct_but_wrong_OR_missing_second"]
            for r in records),
        "teacher_forced_correct_BPE_top1_positions":sum(
            s["native_target_is_top1"] for s in all_golds),
        "total_teacher_forced_full_answer_BPE_plus_EOS_positions":
            len(all_golds),
        "mean_teacher_forced_gold_native_target_CE":
            sum(s["native_target_CE"] for s in all_golds)/len(all_golds),
        "mean_first_divergence_position_1based":
            sum(i for i in firsts if i is not None)/
            max(1,sum(i is not None for i in firsts)),
        "owned_generated_sequences_containing_visible_replacement_UFFFD":
            sum(r["actual_own_final_UTF8_native_byte_status"][
                "replacement_UFFFD_visible_if_rendered_NOW"]
                for r in records),
        "original_physical_native_decoder_forwards":sum(
            r["total_original_S142_Core12_full_native_decoder_forwards"]
            for r in records),
        "already_spent_author_DEV_not_independent_or_blind":True,
    }


def run(s93,side,s142,q2d_dir,q2f_dir,q2i1_dir,q2i3_dir,out):
    check(not out.exists(),"do not overwrite any previous research artifact")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    old_meta=author_provenance()
    old_dev=authored_rows("DEV")
    selected=[x for x in old_dev
              if int(x["id"].rsplit("_",1)[-1])==6]
    check(len(selected)==SOURCE_COUNT
          and Counter(x["kind"] for x in selected)=={k:1 for k in KINDS},
          "strictly 6 ALREADY SCORED DEV source prompts no new EVAL labels")
    q2d=load_q2d_proof(
        q2d_dir/Q2D_REPORT,q2d_dir/Q2D_CTX,q2d_dir/Q2D_LOCAL,s93,s142)
    q2f=json.loads((q2f_dir/Q2F_REPORT).read_text(encoding="utf8"))
    q2i1=json.loads((q2i1_dir/Q2I1_REPORT).read_text(encoding="utf8"))
    q2i3_path=q2i3_dir/Q2I3_REPORT
    q2i3=json.loads(q2i3_path.read_text(encoding="utf8"))
    check(q2i1["stage"]==Q2I1_STAGE
          and q2i1["original_author_new_Russian_source_provenance"]==old_meta
          and q2i1["original_S142_43M_GT_SHA256"]==sha(s142)
          and q2i1["original_S93_SHA256"]==sha(s93)
          and q2i3["stage"]==Q2I3_STAGE
          and q2i3["prior_q2i1_real_joint_report_sha256"]==
              sha(q2i1_dir/Q2I1_REPORT)
          and q2i3["original_S142_43M_GT_sha256"]==sha(s142)
          and q2i3["original_S93_sha256"]==sha(s93)
          and q2i3["source_original_ru_TRAIN_sha256"]==old_meta["TRAIN_sha256"]
          and q2i3["source_prior_scored_ru_DEV_sha256"]==old_meta["DEV_sha256"]
          and q2i3["total_real_original_decoder_forwards"]==988
          and q2i3["identical_HEAD_GAIN_baseline_recurrent_nocarry"]==.25
          and not q2i3["candidate_accepted"]
          and q2i3["output_gate"]==0,
          "source SHA + original 988 physical q2i3 decoder trace pinned")
    base_q2d,_=load_q2d_trained_child(
        q2d_dir/Q2D_CTX,q2d["new_research_checkpoint_contextual"])
    base_q2f=load_physical_q2f_child(
        q2f_dir/Q2F_CTX,
        q2f["projector_contextual_child_checkpoint"],base_q2d)
    base=load_q2i1_physical_context(
        q2i1_dir/Q2I1_CTX,
        q2i1["learned_checkpoint_fp16"]["joint_contextual"],base_q2f)
    children={
        "GT_ZERO_BASELINE":base,
        "GT_RECURRENT":check_trained_joint(
            q2i3_dir/RECURRENT_CKPT,
            q2i3["research_only_updated_FULL_neural_child_FP16"][
                "GT_RECURRENT"],base),
        "GT_NOCARRY_LOCAL":check_trained_joint(
            q2i3_dir/LOCAL_CKPT,
            q2i3["research_only_updated_FULL_neural_child_FP16"][
                "GT_NOCARRY_LOCAL"],base),
    }
    gt_sidecars={
        "GT_RECURRENT":verified_gt12_gates(
            q2i3_dir/RECURRENT_GATES,
            q2i3["research_only_updated_GT12_gate_FP16_sidecar"][
                "GT_RECURRENT"]),
        "GT_NOCARRY_LOCAL":verified_gt12_gates(
            q2i3_dir/LOCAL_GATES,
            q2i3["research_only_updated_GT12_gate_FP16_sidecar"][
                "GT_NOCARRY_LOCAL"]),
    }
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==VOCAB_SIZE and tok.eos_token_id is not None,
          "ORIGINAL S142 49,152 native BPE and EOS, never a task proxy")
    examples=[verified_native_answer(tok,x) for x in selected]
    check(all(x["split"]=="DEV" for x in examples)
          and len(examples)==SOURCE_COUNT,
          "no TRAIN or unopened reserved EVAL data")
    model,sources,bridge,full,expansion=load_actual_original(
        s93,side,s142)
    model.model.add_module("nra3_s119m_onpath",base)
    model.eval()
    frozen_digest=old_original_param_digest(model)
    original_gt={k:v.detach().cpu().clone() for k,v in full.state_dict().items()}
    original_zero=full.gates.detach().clone()
    check(native_full_gt_gate_contract(full,bridge)[
        "original_S142_GT12_12_gates_exact_ZERO"],
        "original pretrained GT12 zero gates must remain unmodified")
    child_snapshots={
        name:{k:v.detach().cpu().clone()
              for k,v in child.state_dict().items()}
        for name,child in children.items()
    }
    measured={};real_calls=0
    try:
        for variant,child in children.items():
            set_gt_route(
                full,variant,
                original_zero if variant=="GT_ZERO_BASELINE"
                else gt_sidecars[variant])
            rows=[]
            original_trace=q2i3["oldDEV_own_native_complete_text"][variant]
            for example in examples:
                previous=next(x for x in original_trace
                              if x["source_id"]==example["id"])
                check(previous["variant"]==variant
                      and previous["teacher_forcing_in_DEV"] is False
                      and previous["original_full_decoder_calls"]==
                          len(previous["generated_own_native_BPE_ids"]),
                      "must reproduce previous actual full q2i3 own tokens")
                outrow=measure_sequence(
                    tok,model,sources,bridge,full,expansion,
                    child,example,previous["generated_own_native_BPE_ids"],
                    variant=variant)
                rows.append(outrow)
                real_calls+=outrow[
                    "total_original_S142_Core12_full_native_decoder_forwards"]
            measured[variant]=rows
    finally:
        with torch.no_grad():
            full.gates.copy_(original_gt["gates"])
            full.mix_logits.copy_(original_gt["mix_logits"])
    check(all(torch.equal(v,full.state_dict()[k].detach().cpu())
              for k,v in original_gt.items())
          and old_original_param_digest(model)==frozen_digest
          and all(all(torch.equal(v,child.state_dict()[k].detach().cpu())
                  for k,v in child_snapshots[name].items())
              for name,child in children.items()),
          "all original pretrained S142 43M+GT and trained FP16 children EXACTLY frozen")
    stats={k:summarize_sequences(v) for k,v in measured.items()}
    check(real_calls==sum(s["original_physical_native_decoder_forwards"]
                          for s in stats.values())
          and SOURCE_COUNT*len(MODELS)<real_calls<=SOURCE_COUNT*len(MODELS)*(
              MAX_SELF_TOKENS+MAX_GOLD_TARGET_TOKENS),
          "true full decoder calls from each real original gold and self prefix")
    result={
        "stage":STAGE,
        "original_S93_SHA256":sha(s93),
        "original_S142_43M_GT_SHA256":sha(s142),
        "previous_q2i3_REAL_Full_native_988_report_SHA256":
            sha(q2i3_path),
        "prior_SOURCE_old_q2i1_DEV_SHA256":old_meta["DEV_sha256"],
        "six_previously_exposed_russian_DEV_source_only":True,
        "source_one_per_each_six_domains":True,
        "three_prior_original_43M_variant_names":list(MODELS),
        "real_original_43M_full_native_decoder_calls":real_calls,
        "full_native_vocab":VOCAB_SIZE,
        "max_own_generated_native_BPE":MAX_SELF_TOKENS,
        "prior_self_native_BPE_sequences_EXACTLY_REPRODUCED":True,
        "correct_gold_native_BPE_used_ONLY_IN_teacher_forced_diagnostic":True,
        "no_CURRENT_OR_FUTURE_gold_native_BPE_IN_self_prefill":True,
        "utf8_native_raw_byte_tail_may_be_incomplete_not_invalid_semantics":True,
        "per_variant_native_full_answer_sequence_summary":stats,
        "per_variant_every_old_source_native_gold_and_own_position":measured,
        "original_43M_GT12_all_neural_weights_unchanged":True,
        "fitted_q2i1_q2i3_neural_FP16_state_dict_unchanged":True,
        "no_train_or_optimizer_or_dev_tuning":True,
        "no_new_independent_BLIND_or_616protected_J7_EOS":True,
        "no_packed_all_in_bpp_or_efficient_KV_CPU_validation":True,
        "no_GOLD_S118_B_spent_EVAL128_J7_labels_or_Jessica_chat":True,
        "no_MAIN_LIVE_Render_Supabase_APK":True,
        "candidate_accepted":False,
        "output_gate":0,
        "decision":"NATIVE_CAUSAL_BPE_BYTE_CONTINUATION_DIAGNOSTIC_NOT_ACCEPTED",
    }
    out.mkdir(parents=True)
    (out/REPORT).write_text(
        json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n",
        encoding="utf8")
    print("S119Q2I5_PHYSICAL_ORIGINAL43M_GOLD_VS_SELF_UTF8_FIRST_DIVERGENCE "+
          json.dumps({
            "real_original_Core12_GT_native_decoder_calls":real_calls,
            "six_previously_scored_source_domains":SOURCE_COUNT,
            "per_variant_native_divergence":stats,
            "native_self_sequences_reproduced_previous_q2i3":True,
            "original_and_fitted_children_immutable":True,
            "candidate_accepted":False,
          },ensure_ascii=False,sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    for name in ("s93","side","s142","q2d_dir","q2f_dir","q2i1_dir",
                 "q2i3_dir","out"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.q2d_dir,a.q2f_dir,
        a.q2i1_dir,a.q2i3_dir,a.out)


if __name__=="__main__":
    main()
