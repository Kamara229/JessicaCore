"""CALIBRI S83: native TRAIN-only diagnostic why arbitrary nouns are not copied.

No checkpoint selection. No model updates. No held-out QA or prompts
(S79/S72/S68/S55/S53/S29 permanently embargoed). This is a measured
TRAIN-only *mechanism diagnosis* and does not imply generalization.

Independent of S82 learned pointer result. Prior model is exact frozen
S74 0.6593 bpp physically paid in GitHub #42562744529. Temporarily
decompress original 36M model to real Apple Silicon MPS FP32 memory;
never save FP32 weights, never touch LIVE or Jessica.

Evaluate 456 authored TRAIN inputs (S30 64, S54 64, history 8,
S69 32, S73 32, S82 256) under teacher-forced gold answers:
 A) gold first token rank vs actual native argmax;
 B) gold noun and gold EOS logit margins vs EOS and real top rival;
 C) how many gold noun IDs actually occur among eligible prompt tokens;
 D) what fraction of already-correct first colors face EOS/unknown noun
    vs simple wrong-noun predictions at second-token step.
This discriminates an EOS-blocking bottleneck from pointer-tokenization
mismatch, using full real vocabulary logits and real 36M frozen GPU.
Only pre-authored TRAIN prompts are read; no independent QA.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s8.broad_replay import SOURCE_BYTES,SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from calibri2.s69 import train_new_contexts_without_old_skills_loss as s69
from calibri2.s73 import first_token_neural_recovery_protected_train as s73
from calibri2.s78 import mps_36m_feature_cached_late_rank16_restoration as s78
from calibri2.s82 import real_mps_generic_token_context_copy as s82
from v186.pairs import build_pairs

def sha(payload):return hashlib.sha256(payload).hexdigest()

def to_plain(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
    return x

def trim(values):
    if not values:return {"count":0}
    sorted_v=sorted(values)
    n=len(values)
    return {"count":n,"median":statistics.median(values),
        "p10":sorted_v[int(.1*(n-1))],"p90":sorted_v[int(.9*(n-1))],
        "min":min(values),"max":max(values)}

@torch.inference_mode()
def teacher_forced_native(dense,head,first,later,tok,prompt,answer):
    gold=tok.encode(answer,add_special_tokens=False)
    if not (1<=len(gold)<=3):raise AssertionError("TRAIN gold answer unusual tokenization")
    prefix=tok.apply_chat_template([{"role":"user","content":prompt}],
        tokenize=False,add_generation_prompt=True)
    prompt_ids=tok.encode(prefix,add_special_tokens=False)
    model_ids=prompt_ids+gold
    ids=torch.tensor([model_ids],dtype=torch.long,device="mps")
    r=dense(input_ids=ids,use_cache=False,output_hidden_states=True)
    h=r.hidden_states[-1][0]
    base=r.logits[0]
    if h.device.type!="mps":raise AssertionError("S83 true GPU hidden required")
    summary=[]
    for step,token in enumerate(gold+[tok.eos_token_id]):
        index=len(prompt_ids)-1+step
        logits=base[index]+F.linear(first(h[index]) if step==0 else later(h[index]),head)
        top_id=int(logits.argmax())
        gold_id=int(token)
        correct=top_id==gold_id
        top_value=float(logits[top_id])
        gold_value=float(logits[gold_id])
        eos_value=float(logits[tok.eos_token_id])
        topk=logits.topk(12)
        # Every diagnostic is TRAIN-only, detached; no gradients.
        summary.append({
            "step":step,
            "gold_id":gold_id,"actual_top_id":top_id,
            "gold_correct":correct,
            "gold_minus_top_margin":gold_value-top_value,
            "gold_minus_EOS_margin":gold_value-eos_value,
            "EOS_top":top_id==tok.eos_token_id,
            "gold_in_top12":gold_id in topk.indices.tolist(),
            "actual_top_in_prompt":top_id in prompt_ids,
        })
    del r
    return {"prompt_tokens":prompt_ids,"gold":gold,"steps":summary}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("S45_frozen_original_CB59",type=Path)
    p.add_argument("S74_frozen_controller",type=Path)
    p.add_argument("S74_physically_paid_manifest",type=Path)
    p.add_argument("S61_real_compressed_source",type=Path)
    p.add_argument("output_dir",type=Path)
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    if not torch.backends.mps.is_available():
        raise RuntimeError("S83 real MPS GPU mandatory; no CPU fallback")
    torch.manual_seed(830083);torch.set_num_threads(2)
    raw=a.S45_frozen_original_CB59.read_bytes()
    side=a.S74_frozen_controller.read_bytes()
    compress=a.S61_real_compressed_source.read_bytes()
    manifest=json.loads(a.S74_physically_paid_manifest.read_text())
    if (len(raw)!=SOURCE_BYTES or sha(raw)!=s65.S45_SHA or
        manifest["S74_GPU_full_controller_SHA256"]!=sha(side) or
        manifest["S61_unchanged_exact_compressed_source_SHA256"]!=sha(compress) or
        manifest["physical_all_inclusive_bytes"]!=len(compress)+len(side)+a.S74_physically_paid_manifest.stat().st_size or
        float(manifest["physical_all_inclusive_bpp"])>.7):
        raise AssertionError("S83 refuses nonfrozen/nonphysically-paid source")
    w=load_file(str(a.S74_frozen_controller))
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new,words=s82.newly_authored_train(tok)
    groups={
       "original":s30.make_training_pairs(tok),
       "augmented":s54.build_new_train(tok),
       "history":build_pairs(tok,"TRAIN")[:4],
       "S69":s69.build_new_train(tok),
       "S73":s73.new_s73_train(tok),
       "S82_new":new,
    }
    if [len(v)*2 for v in groups.values()]!=[64,64,8,32,32,256]:
        raise AssertionError("S83 must use only original fully authored 456 TRAIN")
    digest=sha(json.dumps([
        {"g":g,"q":[p[x]["question"] for x in ("left","right")]}
        for g,ps in groups.items() for p in ps],sort_keys=True).encode())
    dense=s30.dense_from_paid(CB59(raw)).to("mps").eval().requires_grad_(False)
    if sum(t.numel() for t in dense.parameters())!=SOURCE_PARAMS:
        raise AssertionError("S83 temporary full model is not correct 36M")
    first,later=s78.controllers(w)
    first=first.to("mps").eval().requires_grad_(False)
    later=later.to("mps").eval().requires_grad_(False)
    head=dense.lm_head.weight.detach()
    if head.device.type!="mps":raise AssertionError("S83 GPU head not on MPS")
    stats={}
    for group,pairs in groups.items():
        rows=[]
        for pair in pairs:
            for side in ("left","right"):
                q=pair[side]["question"]
                answer=pair[side]["answer"]
                r=teacher_forced_native(dense,head,first,later,tok,q,answer)
                gold=r["gold"];steps=r["steps"];ids=r["prompt_tokens"]
                # Token IDs depend on whitespace segmentation; measure
                # whether an EXACT output noun token could be copied.
                word=gold[1] if len(gold)>1 else None
                candidates=s82.candidate_positions(tok,ids)
                copied=[ids[i] for i in candidates]
                noun_step=steps[1] if len(gold)>1 else None
                rows.append({
                    "pair":pair["pair_id"],"side":side,
                    "first_gold_correct":steps[0]["gold_correct"],
                    "strict_teacher_forced_later_all_gold":
                        all(st["gold_correct"] for st in steps[1:]),
                    "output_token_count":len(gold),
                    "target_second_token_exists_in_prompt":word in ids if word is not None else None,
                    "target_second_token_is_eligible_pointer_position":
                        word in copied if word is not None else None,
                    "native_second_top_is_EOS":noun_step["EOS_top"] if noun_step is not None else None,
                    "native_second_top_id":noun_step["actual_top_id"] if noun_step is not None else None,
                    "native_second_expected_id":word,
                    "native_second_top_is_in_prompt":
                        noun_step["actual_top_in_prompt"] if noun_step is not None else None,
                    "gold_second_minus_top":noun_step["gold_minus_top_margin"] if noun_step is not None else None,
                    "gold_second_minus_EOS":noun_step["gold_minus_EOS_margin"] if noun_step is not None else None,
                    "gold_second_in_full_vocab_top12":noun_step["gold_in_top12"] if noun_step is not None else None,
                    "all_teacher_forced_steps":steps,
                })
        with2=[x for x in rows if x["output_token_count"]>=2]
        firstok=[x for x in with2 if x["first_gold_correct"]]
        stats[group]={
            "total_TRAIN_questions":len(rows),
            "first_token_correct":sum(x["first_gold_correct"] for x in rows),
            "teacher_forced_all_later_gold":sum(x["strict_teacher_forced_later_all_gold"] for x in rows),
            "two_or_more_content_tokens":len(with2),
            "second_token_exists_somewhere_in_input":sum(x["target_second_token_exists_in_prompt"] for x in with2),
            "second_token_copyable_using_S82_eligible_token_filter":sum(x["target_second_token_is_eligible_pointer_position"] for x in with2),
            "if_first_gold_right_then_second_gold_right":sum(x["native_second_top_id"]==x["native_second_expected_id"] for x in firstok),
            "if_first_gold_right_then_second_top_EOS":sum(x["native_second_top_is_EOS"] for x in firstok),
            "if_first_gold_right_then_second_gold_already_in_top12":sum(x["gold_second_in_full_vocab_top12"] for x in firstok),
            "if_first_gold_right_then_second_margin_vs_top":
                trim([x["gold_second_minus_top"] for x in firstok]),
            "if_first_gold_right_then_second_margin_vs_EOS":
                trim([x["gold_second_minus_EOS"] for x in firstok]),
            "real_CPU_or_GPU_gradient_training":False,
        }
        print(json.dumps({"event":"S83_REAL_GPU_FULL_MODEL_CAUSAL_MECHANISM_TRAIN_DIAG",
            "TRAIN_group":group,**stats[group]}),flush=True)
        # Raw TRAIN evidence can be used to propose S84 architecture.
        (a.output_dir/f"S83_TRAIN_ONLY_RAW_{group}.json").write_text(json.dumps(rows,indent=2))
    # Explicitly no updated model/weights, no full response QA.
    report={
       "event":"S83_FINAL_ORIGINAL_FROZEN_S74_TRULY_GPU_COPY_EOS_TRAIN_DIAG",
       "actual_36M_model_on_GPU":True,
       "temporary_decoded_dense_FP32_weights_saved":False,
       "physically_paid_S74_bpp_unchanged":manifest["physical_all_inclusive_bpp"],
       "TRAIN_sample_count":456,
       "S82_new_TRAIN_nouns_authored":words,
       "TRAIN_only_source_questions_sha256":digest,
       "TRAIN_results":stats,
       "NO_model_selection_or_gradients_executed":True,
       "all_inspected_QA_including_S79_permanently_EMBARGOED":True,
       "no_generalization_or_FP32_parity_claim":True,
       "no_LIVE_or_Jessica_chat_changes":True,
    }
    (a.output_dir/"S83_FROZEN_TRAIN_ONLY_GPU_TOKENIZATION_AND_EOS_DIAGNOSTIC.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({"event":"S83_REAL_GPU_TRAIN_ONLY_DIAG_COMPLETE",
        "all_456_TRAIN_questions_covered":True,
        "real_mps_gpu":True,"no_QA_read":True,
        "S82_new_frozen_noun_EOS_diagnostics":stats["S82_new"]}),flush=True)

if __name__=="__main__":main()
