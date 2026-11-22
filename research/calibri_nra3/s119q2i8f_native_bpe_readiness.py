"""S119q2i8f: TRUE pinned native 49152-way ByteLevel BPE+EOS readiness audit.

Read-only PRE-TRAIN gate of Q2I8e 70 synthetic 3+-word Russian targets.
No model weights, optimizer, hidden tests, inference, LIVE or production.
Checks native chat template, exact answer-token roundtrip, full gold EOS,
actual UTF8 incompleteness, strict 160-token original decoder context.
NEVER truncate examples or train on DEV to make readiness pass.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import (
    STAGE as SOURCE_STAGE,rows,verify,KINDS,PER_KIND,
)
from research.calibri_nra3.s119q2i5_real43m_own_vs_gold_utf8_sequence import (
    native_bytelevel_full_utf8_status,
)

STAGE="CALIBRI_S119Q2I8F_PINNED_NATIVE_49152_MULTISENTENCE_READINESS_NO_TRAIN"
MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
REV="12fd25f77366fa6b3b4b768ec3050bf629380bac"
MAX_ANSWER_NATIVE=48 # legacy Q2I1 maximum includes EOS
MAX_ORIGINAL_PREFIX=160
SELECTION_PER_KIND=3
MAX_INPUT_TOKENS=MAX_ORIGINAL_PREFIX-1

def audit_tokenizer(tok,source=None):
    dataset=rows()
    provenance=verify() if source is None else source
    if provenance["stage"]!=SOURCE_STAGE or len(dataset)!=70:
        raise ValueError("source provenance/size changed before native audit")
    if tok.vocab_size!=49152 or tok.eos_token_id is None:
        raise ValueError("only true original native 49152 BPE tokenizer plus EOS")
    records=[]
    for r in dataset:
        prompt=str(r["prompt"])
        full_chat=tok.apply_chat_template(
            [{"role":"user","content":prompt}],
            tokenize=False,add_generation_prompt=True)
        prefix=list(tok.encode(full_chat,add_special_tokens=False))
        joined=list(tok.encode(full_chat+r["completion"],add_special_tokens=False))
        reasons=[]
        if not prefix or joined[:len(prefix)]!=prefix:
            reasons.append("ANSWER_TOKENS_NOT_CAUSAL_CHAT_PREFIX")
            answer=[]
        else:
            answer=joined[len(prefix):]
        decoded=tok.decode(answer,skip_special_tokens=True)
        if decoded!=r["completion"]:
            reasons.append("BYTELEVEL_ANSWER_NOT_EXACT_ROUNDTRIP")
        if not answer or int(tok.eos_token_id) in answer:
            reasons.append("UNEXPECTED_EOS_OR_EMPTY_ANSWER")
        if len(answer)+1>MAX_ANSWER_NATIVE-0:
            reasons.append("ANSWER_LONGER_THAN_LEGACY_48_NATIVE_WITH_EOS")
        if len(prefix)+len(answer)>MAX_INPUT_TOKENS:
            reasons.append("EXCEEDS_ORIGINAL_160_TOKEN_CONTEXT_AT_EOS")
        continuation=0
        states=[]
        for t in range(len(answer)+1):
            state=native_bytelevel_full_utf8_status(tok,answer[:t])[
                "entire_original_native_BPE_prefix_UTF8_status"]
            states.append(state)
            if state=="INCOMPLETE_UTF8_NEEDS_CONTINUATION":
                continuation+=1
        if any(x not in ("VALID_COMPLETE_UTF8","INCOMPLETE_UTF8_NEEDS_CONTINUATION")
               for x in states):
            reasons.append("INVALID_INTERMEDIATE_GOLD_BYTE_SEQUENCE")
        if states and states[-1]!="VALID_COMPLETE_UTF8":
            reasons.append("EOS_WOULD_FOLLOW_INCOMPLETE_UTF8")
        records.append({
            "id":r["id"],"kind":r["kind"],
            "prompt_sha256":hashlib.sha256(prompt.encode("utf8")).hexdigest(),
            "prefix_native":len(prefix),"answer_native":len(answer),
            "full_answer_native_and_EOS":len(answer)+1,
            "context_at_EOS":len(prefix)+len(answer),
            "incomplete_gold_UTF8_prefix_positions":continuation,
            "safe_for_original_native_train":not reasons,
            "failed_hard_contract_reasons":reasons,
        })
    passed=[v for v in records if v["safe_for_original_native_train"]]
    selected=[v for k in KINDS for v in [
        x for x in passed if x["kind"]==k][:SELECTION_PER_KIND]]
    if len(selected)!=len(KINDS)*SELECTION_PER_KIND:
        raise ValueError("not enough native-safe samples for an even 3x7 mini-round")
    if len({v["id"] for v in selected})!=len(selected):
        raise ValueError("selected IDs duplicated")
    return {
        "stage":STAGE,"source_stage":SOURCE_STAGE,
        "TRAIN_source_SHA256":provenance["TRAIN_sha256"],
        "pinned_huggingface_tokenizer":MODEL,
        "pinned_revision":REV,
        "original_native_vocab":tok.vocab_size,
        "real_native_EOS_ID":int(tok.eos_token_id),
        "all_NEW_70_native_audited_count":len(records),
        "pass_original_decoder_limits":len(passed),
        "excluded_original_decoder_limits":len(records)-len(passed),
        "pass_by_kind":dict(Counter(v["kind"] for v in passed)),
        "min_answer_BPE_including_EOS":min(x["full_answer_native_and_EOS"] for x in records),
        "max_answer_BPE_including_EOS":max(x["full_answer_native_and_EOS"] for x in records),
        "max_native_context_at_EOS":max(x["context_at_EOS"] for x in records),
        "gold_incomplete_UTF8_previous_prefix_total":sum(
            x["incomplete_gold_UTF8_prefix_positions"] for x in records),
        "selected_balanced_21_TRAIN_IDS":[v["id"] for v in selected],
        "selected_physical_native_target_BPE_with_EOS":sum(
            v["full_answer_native_and_EOS"] for v in selected),
        "split":"TRAIN_ONLY",
        "no_gold_heldout_DEV_or_616J7_answers_used":True,
        "no_original_models_or_fitted_FP16_loaded_or_changed":True,
        "candidate_accepted":False,"output_gate":0,
        "all_row_metrics":records,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise ValueError("never overwrite research artifact")
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REV)
    report=audit_tokenizer(tok)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("S119Q2I8F_TRUE_NATIVE_READINESS "+json.dumps({
        k:v for k,v in report.items() if k not in ("all_row_metrics","selected_balanced_21_TRAIN_IDS")
    },ensure_ascii=False,sort_keys=True),flush=True)
if __name__=="__main__":
    main()
