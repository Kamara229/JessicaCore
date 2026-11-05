"""CALIBRI v167: measure what the original 135M FP32 ACTUALLY knows.

No teacher KL/NLL/PPL or output distribution matching used to decide quality.
No model training/selection; FP32/CB59/CB161 all weights and archives frozen.
Strict per-question records let us separate original limitations from
additional compression damage and actual generated response speed.
"""
import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE,MAX_SOURCE_BYTES
from v153.codec import unpack,report
from v153.runtime import install_paid_block
from v162.evidence import frozen_v161
from v146.scoring import encode_prompt,label_tokens
from v108.generation import build_chat
from v94.generation import generate_one
from .protocol import ARMS,DOMAINS,PHYSICAL_BYTES,check
from .sources import load_all,generation_probes,digest
from .score import choices,generation,parse_strict

def predicted(logits,token_ids,case):
    if logits.ndim!=1 or not torch.isfinite(logits).all():
        raise ValueError("nonfinite original 135M logits")
    chosen=int(logits[list(token_ids)].argmax())
    return {"index":case.index,"prompt_sha":case.sha,
            "chosen_index":chosen,"chosen_label":case.labels[chosen],
            "gold_label":case.labels[case.gold],
            "correct":chosen==case.gold,
            "pair_id":case.pair_id}

@torch.inference_mode()
def last_logits(model,ids):
    logits=model(input_ids=ids,use_cache=False).logits[0,-1]
    if logits.numel()!=49152:
        raise ValueError("wrong pretrained 135M vocabulary")
    return logits.detach()

def score_qa(student,teacher,tokenizer,source,physical,root):
    token_labels={dom:label_tokens(tokenizer,source[dom][0].labels) for dom in DOMAINS}
    rows={a:{d:[] for d in DOMAINS} for a in ARMS}
    durations={a:0.0 for a in ARMS}
    ids_by_domain={d:[encode_prompt(tokenizer,t.prompt) for t in source[d]]
                   for d in DOMAINS}
    for domain in DOMAINS:
        for c,ids in zip(source[domain],ids_by_domain[domain]):
            start=perf_counter()
            a=last_logits(teacher,ids)
            durations["FP32"]+=perf_counter()-start
            rows["FP32"][domain].append(predicted(a,token_labels[domain],c))
            start=perf_counter()
            b=last_logits(student,ids)
            durations["CB59"]+=perf_counter()-start
            rows["CB59"][domain].append(predicted(b,token_labels[domain],c))
    with install_paid_block(student,physical,root) as installed:
        if installed["paid_2D_modules"]!=7:
            raise ValueError("CB161 is not the seven true original paid operator repair")
        for domain in DOMAINS:
            for c,ids in zip(source[domain],ids_by_domain[domain]):
                start=perf_counter()
                scores=last_logits(student,ids)
                durations["CB161"]+=perf_counter()-start
                rows["CB161"][domain].append(predicted(scores,token_labels[domain],c))
    return rows,durations

def real_generation(student,teacher,tokenizer,physical,root,probes):
    records={arm:[] for arm in ARMS}
    for i,probe in enumerate(probes):
        chat=build_chat(tokenizer,probe["question"])
        # Rotate order to limit warmup/thermal drift. Model versions immutable.
        order=ARMS[i%3:]+ARMS[:i%3]
        for arm in order:
            model=teacher if arm=="FP32" else student
            context=(install_paid_block(student,physical,root) if arm=="CB161"
                     else nullcontext())
            with context:
                result=generate_one(model,tokenizer,chat,stop_at_eos=True)
            ids=result["ids"]
            if not 1<=len(ids)<=32 or result["prompt_SHA"]!=digest(chat):
                raise ValueError("invalid real greedy output or changed prompt")
            records[arm].append({
                "id":probe["id"],"domain":probe["domain"],
                "question":probe["question"],
                "prompt_sha":probe["sha"],
                "actual_chat_prompt_sha":result["prompt_SHA"],
                "gold_reference_if_applicable":probe["gold"],
                "pair_id":probe["pair_id"],
                "verbatim_text":result["text"],
                "actual_generated_token_ids":ids,
                "token_count":len(ids),
                "strict_exact_gold":parse_strict(result["text"],probe["gold"]),
                "eos_generated":result["eos_generated"],
                "distinct_ratio":len(set(ids))/len(ids),
                "full_wall_seconds":result["elapsed_seconds"],
                "prefill_seconds":result["prefill_seconds"],
                "decode_seconds":result["decode_seconds"],
            })
    for arm in ARMS:
        records[arm].sort(key=lambda r:r["id"])
    return records

def run(archive,old_report,seal,output):
    check()
    torch.set_num_threads(2)
    output.mkdir(parents=True,exist_ok=True)
    data=archive.read_bytes()
    physical=unpack(data)
    if len(data)!=PHYSICAL_BYTES or len(data)>MAX_SOURCE_BYTES:
        raise ValueError("incorrect all-in physical source size")
    if hashlib.sha256(physical.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("original immutable CB59 source SHA changed")
    prior=frozen_v161(physical,old_report,seal)
    fixed={"stage":"CALIBRI_v167_FP32_FIRST_BEHAVIORAL_CAPABILITY",
           "frozen_original_source_SHA256":physical.sha256,
           "source_bytes":len(data),"unique_original_parameters":TOTAL_UNIQUE,
           "all_in_bpp":8*len(data)/TOTAL_UNIQUE,
           "unseen_validation_rows":[800,896],
           "synthetic_domains":["Arithmetic","Binding","Logic"],
           "arms":ARMS,"domains":DOMAINS,
           "questions_per_domain":{"SciQ":96,"BoolQ":96,"Arithmetic":32,
                                   "Binding":32,"Logic":32},
           "full_greedy_task_count":48,
           "no_test_labels_loaded_before_source_seal":True,
           "no_KL_NLL_or_PPL_quality_gates":True,
           "original_v89_independent_TEST_NOT_USED":True}
    (output/"v167_frozen_protocol_BEFORE_LOADING_QUESTIONS.json").write_text(
        json.dumps(fixed,ensure_ascii=False,indent=2)+"\n")
    source,provenance=load_all()
    probes=generation_probes()
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("original teacher wrong transformer depth")
    with TemporaryDirectory(dir=output) as tmp:
        root=Path(tmp)
        raw=root/"immutable_CB59.cb59"
        raw.write_bytes(physical.original)
        student,metadata,_=make_model("baseline",SimpleNamespace(original=raw,folder=root))
        if not metadata["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("not using original packed model")
        if student.model.embed_tokens.handle is not student.lm_head.handle:
            raise ValueError("tied original source not preserved")
        grade_records,first_token_times=score_qa(student,teacher,tokenizer,source,physical,root)
        choice_summary=choices(source,grade_records)
        free_records=real_generation(student,teacher,tokenizer,physical,root,probes)
        free_summary=generation(probes,free_records)
        if any(p.ndim>=2 for p in student.parameters()):
            raise ValueError("dense original trained FP32 matrix in packed runtime")
    payload={"stage":fixed["stage"],"predeclared":fixed,"source":report(data),
             "frozen_original_v161_evidence":prior,"dataset_and_prompt_provenance":provenance,
             "fp32_first_actual_288_choice_case_records":grade_records,
             "actual_288_choice_answer_accuracy_and_fp32_ceiling":choice_summary,
             "real_48_free_generation_case_records":free_records,
             "actual_free_generation_and_cpu_decode_speed":free_summary,
             "choice_only_forward_cpu_seconds_not_throughput":first_token_times,
             "NO_KL_NLL_PPL_USED":True,
             "FP32_teacher_is_a_135M_INSTRUCT_NOT_AN_IDEAL_ORACLE":True,
             "strict_text_exact_matching_is_not_complete_semantic_grade":True,
             "no_refitting_frozen_models_or_GT_training":True,
             "no_live_Jessica_Render_Supabase_APK_or_main_changes":True,
             "protected_official_independent_TEST_not_opened":True}
    target=output/"v167_original_vs_packed_behavioral_qa_and_full_generation.json"
    target.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":payload["stage"],"all_in_bpp":fixed["all_in_bpp"],
        "choice_accuracy":choice_summary["by_domain"],
        "free_generation":free_summary["by_arm"],
        "report":str(target)},ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("frozen_complete_CB161",type=Path)
    p.add_argument("completed_v161_source_report",type=Path)
    p.add_argument("frozen_v161_seal",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.frozen_complete_CB161,a.completed_v161_source_report,a.frozen_v161_seal,a.out)
