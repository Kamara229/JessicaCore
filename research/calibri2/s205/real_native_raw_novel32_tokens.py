"""S205 real, read-only full CPU J7 replay with raw 32 novel TRAIN token capture.

The wrapper observes the *unaltered* S190 true native inference outputs. It
does not select a route, substitute a token, alter scores, or inspect gold
until all 32 answers have been generated. Replays the pinned S204 real 616
CPU J7 with the exact same physical unaccepted FP16 candidate.

This is research-only diagnosis. It CANNOT accept model weights or change
LIVE Jessica, Render, Supabase, the original cohort, or protected QA/TEST.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from calibri2.s190 import real_cpu_new_train_paraphrase_probe as s190
from calibri2.s160 import real_parallel_slots_j7 as s160
from calibri2.s204 import real_protected_candidate_cpu_j7 as s204

STAGE = "S205_REAL_NATIVE_CPU_J7_RAW_32_NOVEL_TRAIN_TOKEN_TRACE"
S204_IMMUTABLE_RUN = 43389569329


def hash_q(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()


def verify_and_analyze_raw(raw, *, frozen_novel, source_flags, s204_full, eos_id):
    """Pure posthoc validation AFTER native decoding, never at inference."""
    if (len(raw)!=32 or len(frozen_novel)!=32
        or len({hash_q(q) for q,_,_ in frozen_novel})!=32
        or source_flags.get("real_CPU_full_J7_new_TRAIN_all32") is not True
        or source_flags.get("no_gold_or_SHA_at_inference_or_routing") is not True
        or s204_full.get("stage")!="S204_ACTUAL_NATIVE_616_CPU_J7_PROTECTED_NRA_CANDIDATE"
        or s204_full.get("full_456_correct_EOS")!=456
        or s204_full.get("full_128_correct_EOS")!=128
        or s204_full.get("model_accepted") is not False):
        raise AssertionError("S205 must preserve unaccepted 616 native CPU parent")

    # This dictionary is used strictly AFTER ALL 32 native predictions exist;
    # it never enters original routing, J7 decoding, selection or training.
    predeclared={hash_q(q):(q,expected,group) for q,expected,group in frozen_novel}
    flagged={r["prompt_sha256"]:r for r in source_flags["hashed_per_prompt_audit"]}
    if len(flagged)!=32 or set(flagged)!=set(predeclared):
        raise AssertionError("S205 frozen 32 TRAIN source hashset not identical")

    seen=set()
    rows=[]
    group_stats={}
    for item in raw:
        question=item.get("question")
        predicted=item.get("generated_ids")
        if (not isinstance(question,str)
            or not isinstance(predicted,list) or not predicted
            or len(predicted)>4 or any(type(x) is not int or x<0 for x in predicted)):
            raise AssertionError("S205 malformed actual native CPU token transcript")
        identity=hash_q(question)
        if identity not in predeclared or identity in seen:
            raise AssertionError("S205 native CPU decoded a duplicate/uncommitted TRAIN prompt")
        seen.add(identity)
        frozen_q, gold_text,group=predeclared[identity]
        if question!=frozen_q:
            raise AssertionError("S205 original immutable prompt changed")
        expected=item["expected_tokens_fn"](gold_text) if False else None
        # Actual tokenizer invoked in caller when constructing expected_ids:
        expected_ids=item.get("posthoc_expected_ids")
        if (not isinstance(expected_ids,list) or
            any(type(x) is not int for x in expected_ids) or
            expected_ids[-1]!=eos_id):
            raise AssertionError("S205 posthoc target must be tokenized by frozen model")
        target_len=2 if group=="numeric_new_wording" else 3
        if len(expected_ids)!=target_len:
            raise AssertionError("S205 tokenizer original TRAIN target length drift")
        first_ok=predicted[0]==expected_ids[0]
        full_ok=predicted==expected_ids
        eos_ok=predicted[-1]==eos_id
        flags=flagged[identity]
        if (flags["group"]!=group
            or flags["generated_tokens"]!=len(predicted)
            or flags["first_token_correct"]!=first_ok
            or flags["full_EOS_correct"]!=full_ok
            or flags["eos_present"]!=eos_ok):
            raise AssertionError("S205 raw native token trace contradicts S204 J7 transcript flags")
        diffs=[
            pos for pos in range(max(len(predicted),len(expected_ids)))
            if (predicted[pos] if pos<len(predicted) else None)!=
               (expected_ids[pos] if pos<len(expected_ids) else None)
        ]
        group_stats.setdefault(group,{"count":0,"first_correct":0,
                                      "second_correct":0,"EOS_correct":0,
                                      "full_correct":0})
        summary=group_stats[group]
        summary["count"]+=1
        summary["first_correct"]+=int(first_ok)
        summary["second_correct"]+=int(
            len(predicted)>1 and len(expected_ids)>1
            and predicted[1]==expected_ids[1])
        summary["EOS_correct"]+=int(eos_ok)
        summary["full_correct"]+=int(full_ok)
        rows.append({
            "question_sha256":identity,
            "source_group":group,
            "expected_token_ids_posthoc":expected_ids,
            "actual_generated_token_ids_native_CPU":predicted,
            "first_token_correct":first_ok,
            "second_token_correct":len(predicted)>1
                and len(expected_ids)>1 and predicted[1]==expected_ids[1],
            "EOS_correct":eos_ok,
            "full_plus_EOS_correct":full_ok,
            "all_divergent_positions":diffs,
            "first_divergent_position":diffs[0] if diffs else None,
        })
    if seen!=set(predeclared) or len(rows)!=32:
        raise AssertionError("S205 incomplete actual native CPU 32 novel source outputs")
    if (sum(row["first_token_correct"] for row in rows)
            !=s204_full["32_NEW_TRAIN_first_correct"]
        or sum(row["full_plus_EOS_correct"] for row in rows)
            !=s204_full["32_NEW_TRAIN_full_EOS_correct"]
        or sum(row["EOS_correct"] for row in rows)!=32):
        raise AssertionError("S205 original S204 native complete 616 results mismatch")
    return {
        "stage":STAGE,
        "pinned_S204_native_CPU_616_run":S204_IMMUTABLE_RUN,
        "old_456_full_EOS_unchanged":456,
        "new_128_full_EOS_unchanged":128,
        "novel32_FIRST_correct":sum(x["first_token_correct"] for x in rows),
        "novel32_FULL_EOS_correct":sum(x["full_plus_EOS_correct"] for x in rows),
        "novel32_eos_correct":sum(x["EOS_correct"] for x in rows),
        "group_token_position_breakdown":group_stats,
        "actual_native_decoded_token_ids_per_SOURCE_TRAIN":sorted(
            rows,key=lambda row:row["question_sha256"]),
        "no_inference_routes_use_gold_or_SHA":True,
        "only_posthoc_original_SOURCE_TRAIN_labels":True,
        "unaccepted_original_FP16_checkpt_persisted":True,
        "S204_bpp_and_independent_speed_unresolved":True,
        "QA_TEST_not_used":True,
        "model_accepted":False,
        "LIVE_Jessica_Render_Supabase_untouched":True,
    }


def main():
    if len(sys.argv)<17 or sys.argv[1]!="1":
        raise SystemExit("S205 requires identical original S204 1 + 616 native J7 CLI")
    out=Path(sys.argv[-1])
    if out.exists():
        raise FileExistsError("S205 requires fresh isolated native J7 output directory")
    native=s190.probe_true_frozen_cpu
    raw=[]
    calls=[0]
    def traced_probe(*args,**kwargs):
        if calls[0]!=0:
            raise AssertionError("S205 native 32 TRAIN probe must execute exactly once")
        calls[0]+=1
        original=s160.cached_full_cpu_parallel
        def observe(*a,**kw):
            # Observe every decoder call; never match source or expected at
            # inference. The original frozen model executes UNCHANGED.
            result=original(*a,**kw)
            q=a[6] if len(a)>6 else kw.get("question")
            if not isinstance(q,str) or len(result)!=2:
                raise AssertionError("S205 original real CPU decoder ABI changed")
            ids=list(result[0])
            if any(type(z) is not int for z in ids):
                raise AssertionError("S205 raw CPU IDs must be native int tokens")
            raw.append({"question":q,"generated_ids":ids})
            return result
        s160.cached_full_cpu_parallel=observe
        try:
            return native(*args,**kwargs)
        finally:
            s160.cached_full_cpu_parallel=original
    s190.probe_true_frozen_cpu=traced_probe
    try:
        s204.main()  # FULL actual CPU 456+128+32, read-only candidate.
    finally:
        s190.probe_true_frozen_cpu=native
    if calls!=[1] or len(raw)!=32:
        raise AssertionError("S205 genuine original S190 32 native CPU decodes missing")
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL, MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    frozen=s190.new_train_paraphrase_probes()
    mapping={hash_q(q):tok.encode(answer,add_special_tokens=False)+[tok.eos_token_id]
             for q,answer,_ in frozen}
    # Construct expected IDs ONLY AFTER all native generated token IDs exist.
    for row in raw:
        identity=hash_q(row["question"])
        if identity not in mapping:
            raise AssertionError("S205 decoded question not in precommitted novel TRAIN")
        row["posthoc_expected_ids"]=mapping[identity]
    source=json.loads((out/"S190_REAL_CPU_32_NEW_TRAIN_PARAPHRASE_PROBES.json").read_text())
    full=json.loads((out/"S204_REAL_616_CPU_J7.json").read_text())
    report=verify_and_analyze_raw(
        raw,frozen_novel=frozen,source_flags=source,
        s204_full=full,eos_id=tok.eos_token_id)
    (out/"S205_REAL_CPU_NOVEL32_RAW_TOKEN_POSTHOC.json").write_text(
        json.dumps(report,sort_keys=True,indent=2))
    print(json.dumps({
        "event":STAGE,
        "all_original_456_EOS":456,
        "all_new_128_EOS":128,
        "actual_32_native_full_EOS":report["novel32_FULL_EOS_correct"],
        "group_token_position_breakdown":report["group_token_position_breakdown"],
        "model_accepted":False,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
