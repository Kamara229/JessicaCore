"""Full original trained 135M, TRAIN-only causal operator experiment.

Counterfactual FP32 oracles are NOT compressed source candidates.
No validation/TEST-guided weight selection or model modification.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from paid_lowrank_v76 import get_tokens, WINDOW, make_model
from v84.sources import original_teacher, TARGETS
from v85.physical import CB85
from v85.native_ops import insert_native_adapters, restore_native_adapters
from v86.provenance import freeze_audit
from .oracle import intervene, SCENARIOS
from .metrics import score, mean

CORPORA = ("wiki_valid", "news_valid")
WINDOW_INDICES = (2, 3)
PROMPTS = ("A simple example of this principle is",
           "One afternoon the researchers noticed")

def train_sources():
    result={}
    for domain in CORPORA:
        ids, origin = get_tokens(domain)
        if origin.get("split") != "train":
            raise AssertionError("original official TEST must never be loaded")
        windows=[]
        for ix in WINDOW_INDICES:
            tokens=ids[ix*WINDOW:(ix+1)*WINDOW].clone()
            if tokens.numel()!=WINDOW:
                raise AssertionError("missing original TRAIN source window")
            windows.append({"index":ix, "ids":tokens,
                            "sha":hashlib.sha256(tokens.numpy().tobytes()).hexdigest()})
        result[domain]={"source":origin,"windows":windows}
    return result

@torch.inference_mode()
def greedy(model, tokenizer):
    items=[]
    for prompt in PROMPTS:
        x=tokenizer(prompt,add_special_tokens=False,return_tensors="pt").input_ids
        y=model(input_ids=x,use_cache=True)
        cache=y.past_key_values
        token=y.logits[:,-1].argmax(-1,keepdim=True)
        emitted=[int(token.item())]
        for _ in range(15):
            y=model(input_ids=token,past_key_values=cache,use_cache=True)
            cache=y.past_key_values
            token=y.logits[:,-1].argmax(-1,keepdim=True)
            emitted.append(int(token.item()))
        items.append({"prompt_sha":hashlib.sha256(prompt.encode()).hexdigest(),
                      "distinct_ratio":len(set(emitted))/len(emitted),
                      "max_token_share":max(emitted.count(t) for t in set(emitted))/len(emitted),
                      "tokens":len(emitted),
                      "generated_sha":hashlib.sha256(torch.tensor(
                          emitted,dtype=torch.int64).numpy().tobytes()).hexdigest()})
    return {"mean_distinct_ratio":sum(x["distinct_ratio"] for x in items)/len(items),
            "real_KV_cache":True,"details":items}

@torch.inference_mode()
def evaluate(model, data, refs, tokenizer):
    per_domain={}
    for domain in CORPORA:
        rows=[]
        for window,reference in zip(data[domain]["windows"],refs[domain]):
            ids=window["ids"]
            logits=model(input_ids=ids[None],use_cache=False).logits[0]
            rows.append(score(logits,reference,ids))
        per_domain[domain]=mean(rows)
    return {"real_original_TRAIN_full_model":per_domain,
            "real_CPU_KV_cached_greedy":greedy(model,tokenizer)}

def run(original_path:Path, adapted_path:Path, out:Path):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    original=original_path.read_bytes()
    adapted=adapted_path.read_bytes()
    audit=freeze_audit(original,adapted)
    data=train_sources()
    teacher=original_teacher()
    student,info,_=make_model("baseline",SimpleNamespace(original=original_path,folder=out))
    if not info["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("trained dense source leaked into packed control")
    if info["complete_physical_SHA256"]!=hashlib.sha256(original).hexdigest():
        raise AssertionError("wrong original native compressed source")
    refs={}
    with torch.inference_mode():
        for domain in CORPORA:
            refs[domain]=[teacher(input_ids=w["ids"][None],use_cache=False)
                          .logits[0].detach().clone() for w in data[domain]["windows"]]
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    results={}
    for scenario,paths in SCENARIOS.items():
        with intervene(student,teacher,paths):
            results[scenario]=evaluate(student,data,refs,tok)
        print(json.dumps({"stage":"v87_REAL_oracle_TRAIN_diagnostic",
                          "scenario":scenario,
                          "KL":{d:results[scenario]["real_original_TRAIN_full_model"][d][
                              "full_vocab_KL"] for d in CORPORA}}),flush=True)
    source=CB85(adapted)
    if set(source.extra)!=set(TARGETS):
        raise AssertionError("unexpected frozen dynamic source")
    old=insert_native_adapters(student,source,TARGETS)
    try:
        results["frozen_CB85_no_retraining"]=evaluate(student,data,refs,tok)
    finally:
        restore_native_adapters(old)
    control=results["sham_baseline"]["real_original_TRAIN_full_model"]
    changes={}
    for name,row in results.items():
        changes[name]={domain:{
            "delta_KL_nats":row["real_original_TRAIN_full_model"][domain][
                "full_vocab_KL"]-control[domain]["full_vocab_KL"],
            "delta_NLL_nats":row["real_original_TRAIN_full_model"][domain][
                "next_token_NLL"]-control[domain]["next_token_NLL"]}
            for domain in CORPORA}
    if (original_path.read_bytes()!=original or adapted_path.read_bytes()!=adapted):
        raise AssertionError("original frozen model source modified")
    report={
        "study":"CALIBRI_v87_genuine_TRAIN_causal_FP32_operator_oracle",
        "model":MODEL,"pinned_revision":MODEL_SHA,"frozen_physical_sources":audit,
        "TRAIN_only_origin":{d:{"source":data[d]["source"],
           "window_indices":list(WINDOW_INDICES),
           "token_sha256":[x["sha"] for x in data[d]["windows"]]} for d in CORPORA},
        "experimental_interventions":results,
        "relative_to_CB59_original_packed":changes,
        "IMPORTANT_FP32_oracles_are_uncompressed_diagnostics_NOT_0p7bpw_sources":True,
        "v86_official_unseen_TEST_never_loaded_or_used_for_training":True,
        "no_new_weights_trained_or_source_selected":True,
        "main_live_Jessica_Render_Supabase_APK_untouched":True,
        "caveat":"Only 126 scored next-token positions/domain, original TRAIN. "
          "FP32 oracle interventions reveal potential causal leverage, not "
          "a compact deployable model; 2 greedy prompts are small and unstable."
    }
    target=out/"v87_original_TRAIN_FP32_oracle_diagnostics.json"
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v87_TRAIN_only_finished",
                      "delta_KL_by_scenario":{k:{d:round(x["delta_KL_nats"],5)
                            for d,x in v.items()} for k,v in changes.items()},
                      "real_greedy_distinct":{k:round(
                           v["real_CPU_KV_cached_greedy"]["mean_distinct_ratio"],4)
                           for k,v in results.items()},
                      "scientific_status":"diagnostic only; no new compact source"}),
          flush=True)
    return report
