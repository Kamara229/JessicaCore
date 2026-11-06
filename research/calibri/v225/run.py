"""CALIBRI v225: isolated TRAIN-only logit causes of five v220 repetitions.

Input: original immutable 48-prompt TRAIN rollout in v222 and v224
repeat-location audit. Compare actual FP32/v206/v220 full-vocabulary
logits at positions 3 and 4, conditioning on EACH already-generated
packed trajectory separately. Distinguish changed logits from changed
prefix/feedback. Source physical archives remain immutable and SHA-
authenticated. No training, tuning, DEV, VALID, TEST or Jessica tests.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v84.sources import original_teacher
from v174.run import save
from v176.operators import clear
from v196 import protocol as P
from v196.data import training_inputs
from v196.recover import build_trainable_from_paid_source
from v212.run import SOURCE_V206_SHA256
from v221.run import V220_SOURCE_SHA256


def load_checked(path, digest):
    raw = Path(path).read_bytes()
    if len(raw) != P.SOURCE_BYTES or sha256(raw).hexdigest() != digest:
        raise AssertionError("Model is not the original sealed 135M paid source")
    model, paid = build_trainable_from_paid_source(CB59(raw))
    if len(paid) != 211:
        raise AssertionError("Must preserve all 211 original native codebooks")
    return model.cpu().eval(), paid


@torch.inference_mode()
def probe(model, context, prev, expected_generated, gold, tok, paid=None):
    if paid is not None:
        clear(paid)
    logits = model(input_ids=context,use_cache=False).logits[0,-1].float()
    if paid is not None:
        clear(paid)
    if logits.shape != (P.VOCAB_SIZE,) or not torch.isfinite(logits).all():
        raise AssertionError("Nonfinite or incomplete model logits")
    if any(not 0 <= x < P.VOCAB_SIZE for x in (prev,expected_generated,gold)):
        raise AssertionError("Invalid target or repeated token")
    actual = int(logits.argmax())
    nonrepeat = logits.clone()
    nonrepeat[prev] = float("-inf")
    best_other_id = int(nonrepeat.argmax())
    top = torch.topk(logits,5)
    return {
        "argmax_id":actual,
        "argmax_text":tok.decode([actual]),
        "predicted_repetition":actual == prev,
        "previous_token_id":prev,
        "previous_token_text":tok.decode([prev]),
        "original_rollout_token_id":expected_generated,
        "original_rollout_token_text":tok.decode([expected_generated]),
        "gold_at_this_position_id":gold,
        "gold_at_this_position_text":tok.decode([gold]),
        "argmax_is_gold":actual == gold,
        "logit_repeat_minus_best_nonrepeat":float(logits[prev]-nonrepeat[best_other_id]),
        "best_nonrepeat_id":best_other_id,
        "best_nonrepeat_text":tok.decode([best_other_id]),
        "logit_repeat_minus_original_rollout_choice":
            float(logits[prev]-logits[expected_generated]),
        "logit_repeat_minus_gold":float(logits[prev]-logits[gold]),
        "logp_repeat":float(F.log_softmax(logits,dim=-1)[prev]),
        "logp_gold":float(F.log_softmax(logits,dim=-1)[gold]),
        "top5":[{"id":int(i),"text":tok.decode([int(i)]),"logit":float(v)}
                for i,v in zip(top.indices,top.values)],
    }


def read_provenance(rollout_path,audit_path):
    rollout=json.loads(Path(rollout_path).read_text())
    audit=json.loads(Path(audit_path).read_text())
    if not (rollout.get("completed") and rollout.get("public_DEV16_not_reused")
            and rollout.get("never_train_or_update_source")
            and rollout.get("max_new_tokens") == 12):
        raise AssertionError("v222 original TRAIN output missing verification")
    if rollout["source_hashes"] != {
        "v206":SOURCE_V206_SHA256,"v220":V220_SOURCE_SHA256
    }:
        raise AssertionError("Original v222 source hash mismatch")
    if not audit.get("completed") or not audit.get(
        "strictly_frozen_existing_v222_public_TRAIN_only"):
        raise AssertionError("Expected completed isolated v224 repetition audit")
    if audit.get("sources") != {"v206":0,"v220":5}:
        raise AssertionError("Repeat counts inconsistent with v224")
    if audit.get("v220_repeat_position_histogram_1based") != {"3":4,"4":1}:
        raise AssertionError("v224 frozen repeat positions changed")
    if len(audit["v220_repeat_details"]) != 4:
        raise AssertionError("Expected exactly four affected TRAIN prompts")
    rows={(r["pair_id"],r["side"]):r
          for r in rollout["all_48_opposing_TRAIN_rows"]}
    if len(rows)!=48 or any(r["split"]!="TRAIN" for r in rows.values()):
        raise AssertionError("Original 48 opposing TRAIN cases changed")
    return rows,audit["v220_repeat_details"]


def analyze(v206_path,v220_path,rollout_path,audit_path,out):
    torch.manual_seed(225)
    torch.set_num_threads(2)
    rows,repeat_rows=read_provenance(rollout_path,audit_path)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    selected,train_pairs,valid_pairs,_=training_inputs(tok)
    if len(selected)!=20 or len(train_pairs)!=24 or len(valid_pairs)!=P.VALID_PAIRS:
        raise AssertionError("Original TRAIN source protocol drift")
    pairs={(p["pair_id"],side):p[side] for p in train_pairs
           for side in ("left","right") if p.get("split")=="TRAIN"}
    if len(pairs)!=48 or set(rows)!=set(pairs):
        raise AssertionError("TRAIN prompt IDs differ from original 48")
    sources={
        "v206":load_checked(v206_path,SOURCE_V206_SHA256),
        "v220":load_checked(v220_path,V220_SOURCE_SHA256)
    }
    teacher=original_teacher().cpu().eval()
    output=Path(out)
    output.mkdir(parents=True,exist_ok=True)
    save(output/"v225_PRECOMMITTED_TRAIN_REPETITION_CAUSAL_PROBE.json",{
        "read_only":True,"source_v206_sha":SOURCE_V206_SHA256,
        "source_v220_sha":V220_SOURCE_SHA256,
        "FP32_revision":MODEL_SHA,
        "source_rollout":"v222 original 48 TRAIN samples",
        "source_repetitions":"v224 4 affected TRAIN prompts",
        "target_positions_1based":[3,4],
        "no_dev_valid_qa_test":True,
        "no_weights_fit_or_selection":True,
        "do_not_modify_Jessica_LIVE_main_Render_Supabase_APK":True
    })
    details=[]
    for entry in repeat_rows:
        key=(entry["pair_id"],entry["side"])
        row=rows[key]
        if (row["domain"]!=entry["domain"] or
            row["first_diff_position_1based"]!=2):
            raise AssertionError("Original v222/v224 affected row mismatch")
        original=list(row["v206_generated_token_ids"])
        updated=list(row["v220_generated_token_ids"])
        if original!=entry["original_first_12_token_ids"] or updated!=entry[
                "candidate_first_12_token_ids"]:
            raise AssertionError("Rollout tokens mismatch across v222 and v224")
        sample=pairs[key]
        prompt=sample["prompt_ids"].detach().clone().long()[None]
        gold=list(row["gold_token_ids_including_EOS"])
        for loc in entry["positions"]:
            position=loc["position_1based"]
            if position not in (3,4):
                raise AssertionError("Unplanned repeat location outside 3-4")
            previous_updated=int(updated[position-2])
            repeated=int(updated[position-1])
            if repeated!=previous_updated or repeated!=loc["token_id"]:
                raise AssertionError("v220 repeat missing in original trajectory")
            expected_gold=int(gold[position-1]) if len(gold)>=position \
                          else int(tok.eos_token_id)
            report={
                "split":"TRAIN","pair_id":entry["pair_id"],
                "side":entry["side"],"domain":entry["domain"],
                "repeat_position_1based":position,
                "repeated_token_id":repeated,
                "v206_rollout_first_12_token_ids":original,
                "v220_rollout_first_12_token_ids":updated,
                "gold_token_id_at_position":expected_gold,
                "on_v206_generated_context":{},
                "on_v220_generated_context":{},
            }
            for context_source,generated in (
                ("on_v206_generated_context",original),
                ("on_v220_generated_context",updated),
            ):
                if len(generated) < position:
                    # The original native v206 can stop at EOS before a later
                    # position that only exists in the v220 rollout. Never
                    # fabricate tokens beyond its own terminated generation.
                    report[context_source] = {
                        "not_applicable":True,
                        "reason":"native_v206_rollout_terminated_before_position",
                        "generated_token_length":len(generated)}
                    continue
                context=torch.cat(
                    (prompt,prompt.new_tensor([generated[:position-1]])),dim=1)
                prev=int(generated[position-2])
                source_next=int(generated[position-1])
                teacher_score=probe(
                    teacher,context,prev,source_next,expected_gold,tok)
                report[context_source]["FP32"]=teacher_score
                for name,(model,paid) in sources.items():
                    result=probe(model,context,prev,source_next,
                                 expected_gold,tok,paid)
                    if name == ("v206" if context_source.startswith(
                                "on_v206") else "v220"):
                        if result["argmax_id"] != source_next:
                            raise AssertionError(
                                "Original immutable v222 native rollout does "
                                "not match recomputed greedy token for " +
                                f"{name} {key} pos={position}")
                    report[context_source][name]=result
            details.append(report)
            print(json.dumps({
                "stage":"v225_TRAIN_repeat_logit_progress",
                "pair_id":entry["pair_id"],"side":entry["side"],
                "position_1based":position,
                "v220_repeats_on_own_context":report[
                    "on_v220_generated_context"]["v220"][
                        "predicted_repetition"],
                "FP32_repeats_on_v220_context":report[
                    "on_v220_generated_context"]["FP32"][
                        "predicted_repetition"],
                "v206_repeats_on_v220_context":report[
                    "on_v220_generated_context"]["v206"][
                        "predicted_repetition"]
            }),flush=True)
    if len(details)!=5 or Counter(d["repeat_position_1based"] for d in
                                   details)!=Counter({3:4,4:1}):
        raise AssertionError("Did not inspect all five frozen TRAIN repeats")
    counts={
        name:sum(d["on_v220_generated_context"][name][
            "predicted_repetition"] for d in details)
        for name in ("FP32","v206","v220")
    }
    summary={
        "original_v220_repeat_positions":5,
        "original_affected_TRAIN_cases":4,
        "repeat_choice_on_v220_prefix_counts":{
            "FP32":counts["FP32"],"v206":counts["v206"],
            "v220":counts["v220"]},
        "novel_repetition_relative_to_FP32_on_same_prefix":
            5-counts["FP32"],
        "novel_repetition_relative_to_v206_on_same_prefix":
            5-counts["v206"],
        "no_trained_weights_or_quality_claim":True,
        "not_independent_DEV_quality_or_root_cause_proof":True
    }
    save(output/"v225_ACTUAL_5_TRAIN_repeat_positions_FP32_v206_v220_logits.json",{
        "completed":True,"strict_TRAIN_only":True,
        "immutable_original_v206_v220_archive_sha256":{
            "v206":SOURCE_V206_SHA256,"v220":V220_SOURCE_SHA256},
        "details":details,"summary":summary,
        "Jessica_tests_ran":False
    })
    print(json.dumps({"stage":"v225_ACTUAL_TRAIN_repeat_logit_summary",**summary}),
          flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("sealed_v206_cb59",type=Path)
    p.add_argument("sealed_v220_cb59",type=Path)
    p.add_argument("original_v222_output",type=Path)
    p.add_argument("original_v224_repetition_audit",type=Path)
    p.add_argument("output_dir",type=Path)
    a=p.parse_args()
    analyze(a.sealed_v206_cb59,a.sealed_v220_cb59,a.original_v222_output,
            a.original_v224_repetition_audit,a.output_dir)
