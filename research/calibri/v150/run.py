"""Frozen CB149 QA + genuine autoregressive response check, GitHub research.

The rank-one context adapter learned once in v149 from source TRAIN FIT,
then passed a NEW 2-domain true-NLL VALID check; next we challenge it on
previously untouched *official validation split* rows [600:624] (SciQ and
BoolQ, 24 each), not the locked independent v89 TEST. We also record exact
full-model 32-token greedy response samples on 6 predeclared historical
prompts for qualitative review. No answer labels used as model inputs or fit.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE
from v141.run import ORIGINAL_V140_SHA
from v143.runtime import PackedFusedHead
from v147.runtime import fused_half_head
from v149.codec import unpack,report
from v149.runtime import ContextualHead
from v146.scoring import label_tokens,encode_prompt
from v148.scoring import make_score
from v94.generation import NEW_PROMPTS,generate_one
from v108.generation import build_chat

from .protocol import (CB149_SHA,CB147_SHA,DATASETS,ARMS,GENERATION_INDICES,
                       GENERATION_ARMS,QUESTION_COUNT,check_protocol)
from .sources import load_new_valid_only
from .audit import summarize


@torch.inference_mode()
def teacher_one(teacher,ids):
    output=teacher(input_ids=ids,use_cache=False).logits[0,-1].detach()
    if output.shape!=(49152,):
        raise ValueError("FP32 teacher returned wrong frozen vocab")
    return output


@torch.inference_mode()
def native_hidden(model,ids):
    h=model.model(input_ids=ids,use_cache=False).last_hidden_state
    if h.shape[:2]!=ids.shape or h.shape[2]!=576:
        raise ValueError("native fully packed model decoder hidden wrong shape")
    return h[:,-1:,:].contiguous()


@torch.inference_mode()
def native_logits(head,hidden):
    logits=head(hidden)[0,-1].detach()
    if logits.shape!=(49152,) or not logits.isfinite().all():
        raise ValueError("native full output vocabulary is invalid")
    return logits


def generation_examples(model,tok,prior,context):
    records={arm:[] for arm in GENERATION_ARMS}
    for i,idx in enumerate(GENERATION_INDICES):
        category,text=NEW_PROMPTS[idx]
        chat=build_chat(tok,text)
        arms=GENERATION_ARMS if i%2==0 else GENERATION_ARMS[::-1]
        for arm in arms:
            model.lm_head=prior if arm=="CB147_prior" else context
            with torch.inference_mode():
                generated=generate_one(model,tok,chat,stop_at_eos=True)
            tokens=generated["ids"]
            records[arm].append({
                "historical_probe_index":idx,
                "category":category,
                "question":text,
                "chat_prompt_SHA":generated["prompt_SHA"],
                "generated_text_verbatim":generated["text"],
                "full_generated_token_ids":tokens,
                "actual_generated_tokens":len(tokens),
                "end_of_turn_EOS":bool(generated["eos_generated"]),
                "distinct_token_ratio":len(set(tokens))/max(len(tokens),1),
                "full_model_CPU_seconds":generated["elapsed_seconds"],
                "no_automatic_semantic_correctness_claim":True,
            })
    all_idx=[r["historical_probe_index"] for r in records[GENERATION_ARMS[0]]]
    for arm in GENERATION_ARMS:
        if [r["historical_probe_index"] for r in records[arm]]!=all_idx:
            raise AssertionError("full generations missing predeclared prompts")
    speeds={}
    for arm in GENERATION_ARMS:
        elapsed=sum(r["full_model_CPU_seconds"] for r in records[arm])
        count=sum(r["actual_generated_tokens"] for r in records[arm])
        speeds[arm]={
            "actual_greedy_tokens":count,
            "full_model_including_prefill_seconds":elapsed,
            "tokens_per_second_rough":count/max(elapsed,1e-9),
            "first_token_EOS_count":sum(
                len(r["full_generated_token_ids"])==1 and r["end_of_turn_EOS"]
                for r in records[arm]),
            "mean_distinct_token_ratio":sum(r["distinct_token_ratio"] for r in records[arm])/len(records[arm]),
        }
    return {
        "raw_full_model_generation_examples":records,
        "observed_speed_by_arm":speeds,
        "CB149_vs_CB147_token_speed_ratio":
            speeds["CB149_contextual"]["tokens_per_second_rough"]/
            max(speeds["CB147_prior"]["tokens_per_second_rough"],1e-9),
        "speed_warning":"Six prompts, CPU scheduling noise; must remeasure full speed and compare with CB59 before release.",
        "requires_human_semantic_grammar_factual_review":True,
    }


def run(archive,out):
    torch.set_num_threads(2)
    check_protocol()
    out.mkdir(parents=True,exist_ok=True)
    data=archive.read_bytes()
    physical=unpack(data)
    if physical.sha256!=CB149_SHA or physical.parent.sha256!=CB147_SHA:
        raise ValueError("wrong frozen physically paid CB149 and parent source")
    if physical.parent.paid.sha256!=ORIGINAL_V140_SHA:
        raise ValueError("wrong physical prepaid parent CB140")
    if hashlib.sha256(physical.parent.paid.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("original pretrained codebook source not frozen")
    tasks,sources=load_new_valid_only()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    candidate_tokens={
        "SciQ":label_tokens(tok,("A","B","C","D")),
        "BoolQ":label_tokens(tok,("Yes","No")),
    }
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("not original SmolLM2-135M pretrained teacher")
    results={arm:{domain:[] for domain in DATASETS} for arm in ARMS}
    prefill_times={arm:0. for arm in ARMS}
    with TemporaryDirectory(dir=out) as temp:
        scratch=Path(temp)
        base=scratch/"frozen_parent_CB59.cb59"
        base.write_bytes(physical.parent.paid.original)
        student,metadata,_=make_model(
            "baseline",SimpleNamespace(original=base,folder=scratch))
        if not metadata["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("original pretrained matrix loaded dense")
        native_emb=student.model.embed_tokens
        cb59=student.lm_head
        half=fused_half_head(student,physical.parent.paid,scratch)
        prior=PackedFusedHead(half.handle,half.kernel,physical.parent.bias)
        contextual=ContextualHead(prior,physical)
        if not (native_emb.handle is cb59.handle is prior.handle is contextual.handle):
            raise ValueError("original tied physical weights forked")
        if any(p.ndim>=2 for p in student.parameters()):
            raise ValueError("trained dense source accidentally resident")

        for domain in DATASETS:
            for task in tasks[domain]:
                ids=encode_prompt(tok,task.prompt)
                t0=perf_counter()
                reference=teacher_one(teacher,ids)
                prefill_times["FP32"]+=perf_counter()-t0
                t0=perf_counter()
                h=native_hidden(student,ids)
                shared=perf_counter()-t0
                # Exactly same hidden state for every compressed condition.
                for arm,head in (
                    ("CB59",cb59),
                    ("CB147_prior",prior),
                    ("CB149_contextual",contextual)):
                    t1=perf_counter()
                    scores=native_logits(head,h)
                    prefill_times[arm]+=perf_counter()-t1+shared/3
                    detail=make_score(scores,candidate_tokens[domain],
                                      task.correct_index)
                    detail.update({
                        "question_row":task.index,"prompt_SHA256":task.prompt_sha,
                        "gold_reference_used_only_after_logits":True,
                        "context_token_length":len(ids[0]),
                    })
                    results[arm][domain].append(detail)
                target=make_score(reference,candidate_tokens[domain],
                                  task.correct_index)
                target.update({
                    "question_row":task.index,"prompt_SHA256":task.prompt_sha,
                    "gold_reference_used_only_after_logits":True,
                    "context_token_length":len(ids[0]),
                })
                results["FP32"][domain].append(target)

        generated=generation_examples(student,tok,prior,contextual)
        student.lm_head=cb59
        del student
    del teacher
    quality=summarize(results)
    report_data={
        "stage":"v150_UNSEEN_CHOICES_AND_REAL_32TOKEN_GENERATION",
        "complete_CB149_SHA256":physical.sha256,
        "all_in_complete_physical_source":report(data),
        "original_teacher_F32_evaluation_only":True,
        "predeclared_four_native_model_arms":list(ARMS),
        "data_provenance_from_validation_only":sources,
        "validation_rows_per_domain":QUESTION_COUNT,
        "previously_separate_validation_slices_200_and_400_not_reused":True,
        "no_correct_answers_used_for_fitting_or_model_context_leak":True,
        "no_official_v89_independent_TEST":True,
        "no_green_tunnel_hidden_weights":True,
        "no_LIVE_Jessica_Render_Supabase_APK_changes":True,
        "raw_choice_results":results,
        "choice_quality":quality,
        "separate_heldout_natural_language_NLL_from_v149_NOT_recomputed":True,
        "native_QA_prefill_and_head_seconds_NOT_greedy_generation_speed":prefill_times,
        "paired_genuine_generation":generated,
        "release_not_justified_by_tiny_validation":True,
    }
    path=out/"v150_frozen_CB149_new_QA_and_generation.json"
    path.write_text(json.dumps(report_data,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":report_data["stage"],
        "physical_bpp":report_data["all_in_complete_physical_source"]["complete_bpp_original_unique"],
        "new_qa_choice_results":quality,
        "actual_generation_performance":generated["observed_speed_by_arm"],
        "generation_ratio":generated["CB149_vs_CB147_token_speed_ratio"],
        "report":str(path),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("frozen_CB149",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.frozen_CB149,a.out)
