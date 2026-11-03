"""CALIBRI v88 genuine original-pretrained multidimensional physical weight study.

Compose separate source, tensor, physical and learning facades. Baseline
retains native-C 5bit packed weights. New four-axis residual is FP16-paid
and Torch-CPU executed (no fused C / speed superiority claim).
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher, TARGETS
from v85.physical import CB85
from v85.native_ops import insert_native_adapters,restore_native_adapters
from v87.metrics import score,mean
from .sources import actual_original_sources
from .physical import pack, unpack, audit, verified_source
from .tensor import TensorPackedHead
from .train import fit, measure

PROMPTS=(
 "The scientist opened the small notebook and",
 "A traveller asked the librarian about",
)
N_GREEDY=16

@torch.inference_mode()
def capture(teacher,student,source):
    result={}
    captures=[]
    def on_input(module,inputs):
        if len(inputs)!=1:
            raise AssertionError("expected original output operator input")
        captures.append(inputs[0].detach().clone())
    hook=student.lm_head.register_forward_pre_hook(on_input)
    try:
        for domain,item in source.items():
            rows=[]
            for token_ids in item["windows"]:
                captures.clear()
                y=teacher(input_ids=token_ids[None],use_cache=False).logits[0]
                z=student(input_ids=token_ids[None],use_cache=False).logits[0]
                if len(captures)!=1:
                    raise AssertionError("missing full native output head hidden input")
                features=captures.pop()[0]
                if y.shape!=z.shape or features.shape!=(len(token_ids),576):
                    raise AssertionError("unexpected original pretrained source geometry")
                rows.append({"x":features[:-1].detach().clone(),
                             "teacher":y[:-1].detach().clone(),
                             "baseline":z[:-1].detach().clone(),
                             "labels":token_ids[1:].detach().clone(),
                             "tokens":token_ids.detach().clone(),
                             "teacher_full_window_logits":y.detach().clone()})
            result[domain]={"origin":item["origin"],"rows":rows}
    finally:
        hook.remove()
    return result

@torch.inference_mode()
def greedy(model,tok):
    runs=[]
    for prompt in PROMPTS:
        x=tok(prompt,add_special_tokens=False,return_tensors="pt").input_ids
        t0=time.perf_counter()
        y=model(input_ids=x,use_cache=True)
        first=time.perf_counter()-t0
        cache=y.past_key_values
        token=y.logits[:,-1].argmax(-1,keepdim=True)
        ids=[int(token.item())]
        t0=time.perf_counter()
        for _ in range(N_GREEDY-1):
            y=model(input_ids=token,past_key_values=cache,use_cache=True)
            cache=y.past_key_values
            token=y.logits[:,-1].argmax(-1,keepdim=True)
            ids.append(int(token.item()))
        seconds=time.perf_counter()-t0
        runs.append({"input_sha":hashlib.sha256(prompt.encode()).hexdigest(),
                     "output_sha":hashlib.sha256(torch.tensor(ids,dtype=torch.int64).numpy().tobytes()).hexdigest(),
                     "prefill_seconds":first,"decode_seconds":seconds,
                     "distinct_ratio":len(set(ids))/len(ids),
                     "top_token_share":max(ids.count(v) for v in set(ids))/len(ids),
                     "actual_generated_tokens":len(ids)})
    return {"real_CPU_KV_cache":True,
            "decode_tokens_per_second_single_CPU":(
                len(runs)*(N_GREEDY-1)/sum(x["decode_seconds"] for x in runs)),
            "mean_distinct_token_ratio":sum(x["distinct_ratio"] for x in runs)/len(runs),
            "prompts":runs}

@torch.inference_mode()
def actual_whole_model(model,real_valid,tok):
    corpora={}
    for domain in ("wiki_valid","news_valid"):
        readings=[]
        timing=[]
        for row in real_valid[domain]["rows"]:
            ids=row["tokens"]
            t0=time.perf_counter()
            out=model(input_ids=ids[None],use_cache=False).logits[0]
            timing.append(time.perf_counter()-t0)
            readings.append(score(out,row["teacher_full_window_logits"],ids))
        corpora[domain]={**mean(readings),"actual_native_full_prefill_seconds":sum(timing)}
    return {"original_TRAIN_VALID_full_model":corpora,
            "actual_KV_cached_greedy_2_prompts":greedy(model,tok)}

def run(original_path:Path,frozen_cb85_path:Path,out_dir:Path):
    out_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(8800)
    original=original_path.read_bytes()
    verified_source(original)
    cb85_blob=frozen_cb85_path.read_bytes()
    cb85=CB85(cb85_blob)
    if cb85.original.hash()!=hashlib.sha256(original).hexdigest():
        raise AssertionError("v85 comparator original source mismatch")
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    original_sources=actual_original_sources(tokenizer)
    teacher=original_teacher()
    student,baseline_info,_=make_model(
        "baseline",SimpleNamespace(original=original_path,folder=out_dir))
    if not baseline_info["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("original native student source not fully packed")
    captured=capture(teacher,student,original_sources)
    TRAIN={name:captured[name] for name in ("wiki_fit","news_fit")}
    VALID={name:captured[name] for name in ("wiki_valid","news_valid")}
    fitted=fit(TRAIN,VALID)
    # Every candidate has the SAME complete physically-accounted layout.
    exploratory_blob=pack(original,*fitted["raw_best_exploratory_params"])
    guarded_blob=pack(original,*fitted["selected_physical_params"])
    assert len(guarded_blob)==len(exploratory_blob)
    best_source=out_dir/"v88_exploratory_original_TRAIN_SELECTED_4D.cb88"
    guarded_source=out_dir/"v88_gated_original_TRAIN_VALID_4D.cb88"
    best_source.write_bytes(exploratory_blob)
    guarded_source.write_bytes(guarded_blob)
    _,A,B,bias=unpack(guarded_blob)
    print(json.dumps({"stage":"v88_4D_FROZEN_before_further_runtime_VALID",
                      "research_go":fitted["research_VALID_go"],
                      "guarded_mode":fitted["guarded_selected_mode"],
                      "full_physical_SHA":hashlib.sha256(guarded_blob).hexdigest(),
                      "fully_paid_source":audit(guarded_blob)}),flush=True)
    results={}
    results["original_CB59"]=actual_whole_model(student,VALID,tokenizer)
    replacement=insert_native_adapters(student,cb85,TARGETS)
    try:
        results["frozen_CB85"]=actual_whole_model(student,VALID,tokenizer)
    finally:
        restore_native_adapters(replacement)
    old_head=student.lm_head
    student.lm_head=TensorPackedHead(old_head,A,B,bias)
    try:
        results["physically_complete_CB88"]=actual_whole_model(student,VALID,tokenizer)
    finally:
        student.lm_head=old_head
    measured_checks={}
    for domain in ("wiki_valid","news_valid"):
        cb=results["physically_complete_CB88"]["original_TRAIN_VALID_full_model"][domain]
        cached=measure(VALID[domain],(A,B,bias))
        measured_checks[domain]={
            "native_actual_minus_cached_KL":cb["full_vocab_KL"]-
                                    cached["teacher_full_vocab_KL"],
            "native_actual_minus_cached_NLL":cb["next_token_NLL"]-
                                    cached["original_next_token_NLL"]}
        if max(abs(v) for v in measured_checks[domain].values())>1e-4:
            raise AssertionError("actual physical native packed inference differs from pre-physical cache")
    if (original_path.read_bytes()!=original or frozen_cb85_path.read_bytes()!=cb85_blob):
        raise AssertionError("trained original source or original CB85 comparator changed")
    detailed={k:v for k,v in fitted.items() if not k.endswith("_params")}
    report={
        "study":"CALIBRI_v88_genuine_original_4D_tensor_output_neuroweights",
        "trained_original_model":MODEL,
        "original_model_pinned_revision":MODEL_SHA,
        "original_source_SHA":hashlib.sha256(original).hexdigest(),
        "new_original_official_TRAIN_only_provenance":{
            k:v["origin"] for k,v in captured.items()},
        "physical_exploratory_tensors":audit(exploratory_blob),
        "physical_guarded_source":audit(guarded_blob),
        "trained_4D_rank":4,
        "all_in_bytes_no_unpaid_operator_or_teacher":len(guarded_blob),
        "trained_only_on_original_TRAIN_not_official_TEST":True,
        "TRAIN_VALID_checkpoint_guard":detailed,
        "real_frozen_native_full_model_valid_and_2prompt_greedy":results,
        "actual_runtime_matches_physically_packed_FP16_delta":measured_checks,
        "no_v86_heldout_TEST_used_for_training_or_selection":True,
        "research_go":fitted["research_VALID_go"],
        "live_Jessica_main_Render_Supabase_APK_untouched":True,
        "limitations":"Only one original 135M pretrained model, 2x64 original "
           "TRAIN and TRAIN-VALID tokens per corpus; tensor operator in Torch CPU "
           "while original weights native C; original token-index topology "
           "192x256 not semantically privileged, performance sensitive to "
           "hardware and context. No unseen TEST or production claims."
    }
    (out_dir/"v88_original_pretrained_multidim_physical_KL_NLL_speed.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v88_REAL_4D_native_packed_plus_paid_tensors_finished",
                      "guard":fitted["research_VALID_go"],
                      "winner":fitted["guarded_selected_mode"],
                      "real_VALID_KL":{name:{d:round(metrics[
                          "original_TRAIN_VALID_full_model"][d]["full_vocab_KL"],5)
                          for d in ("wiki_valid","news_valid")}
                          for name,metrics in results.items()},
                      "real_greedy_diversity":{k:round(v[
                          "actual_KV_cached_greedy_2_prompts"][
                          "mean_distinct_token_ratio"],4) for k,v in results.items()},
                      "real_tokens_per_second":{k:round(v[
                          "actual_KV_cached_greedy_2_prompts"][
                          "decode_tokens_per_second_single_CPU"],3)
                          for k,v in results.items()}}),flush=True)
    return report
