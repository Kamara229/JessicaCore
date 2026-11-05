"""v182: same-input FP32/FP16/decoded-V1/native-V1 first-divergence study.

This is a DIAGNOSTIC, not another compression candidate or a success claim.
Unpaid temporary dense source is deliberately counted ONLY as comparison,
not as deployment storage. No tests go through Jessica.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import unpack5
from v84.sources import original_teacher
from v169.sources import all_cases
from v174.protocol import freeze_cases
from v174.run import save
from v181.codec import CB181
from v181.native import build as build_native
from . import protocol as p


def decode_handle(h):
    """Literal FP32 decoded matrix from the actual paid 5-bit source."""
    if h.group != 1:
        raise ValueError("Only exact frozen V1 oracle is allowed")
    m, n = h.shape
    ids = unpack5(h.packed.tobytes(), m*n)
    if ids.shape != (m*n,) or np.any(ids >= 32):
        raise AssertionError("Corrupt original packed 5-bit V1 IDs")
    source = h.centroids[ids].reshape(m,n) * h.row_scales[:,None]
    if not np.isfinite(source).all():
        raise AssertionError("Nonfinite physically decoded source")
    return np.ascontiguousarray(source, dtype=np.float32)


def validate_teacher_source(model, source):
    actual=model.state_dict()
    desc=source.meta["tensor_descriptors"]
    if (set(actual) != set(source.meta["full_original_state_keys"]) or
            set(desc)!=set(source.meta["trained_unique_keys"])):
        raise ValueError("State list differs from real pretrained source")
    checked=0
    for name,d in desc.items():
        if d["dtype"]=="f4":
            reference=torch.from_numpy(np.array(source.raw[name],copy=True))
            if not torch.equal(actual[name].cpu(),reference):
                raise AssertionError("Uncompressed 1D weights differ from FP32: "+name)
            checked+=1
        elif d["dtype"]!="CB181_K32V1":
            raise AssertionError("Unexpected unverified physical source dtype")
    if checked<1:
        raise AssertionError("No immutable 1D operator weights checked")
    if (model.model.embed_tokens.weight.data_ptr() !=
            model.lm_head.weight.data_ptr()):
        raise AssertionError("Original pretrained tied weights not shared")
    return {"checked_1d_tensors":checked,"source_named_state_tensors":len(desc),
        "original_unique_parameters":source.meta["original_unique_parameter_count"],
        "tied_embedding_head_in_FP32":True}


@torch.no_grad()
def make_dense_variant(reference, archive, arm, decoded_cache):
    if arm not in ("FP16_ROUND","V1_DENSE","V1_HEAD_ONLY","V1_BODY_ONLY"):
        raise ValueError("Unregistered diagnostic arm")
    model=copy.deepcopy(reference).eval()
    state=model.state_dict()
    converted_count=0
    for key,d in archive.meta["tensor_descriptors"].items():
        if d["dtype"]!="CB181_K32V1":
            continue
        is_head=(key=="model.embed_tokens.weight")
        if arm=="V1_HEAD_ONLY" and not is_head:continue
        if arm=="V1_BODY_ONLY" and is_head:continue
        if arm=="FP16_ROUND":
            state[key].copy_(state[key].to(torch.float16).to(torch.float32))
        else:
            with torch.no_grad():
                state[key].copy_(torch.from_numpy(decoded_cache[key]))
        converted_count+=1
    expected={"FP16_ROUND":211,"V1_DENSE":211,
              "V1_HEAD_ONLY":1,"V1_BODY_ONLY":210}[arm]
    if converted_count!=expected:
        raise AssertionError(f"{arm}: replaced {converted_count}, expected {expected}")
    if model.model.embed_tokens.weight.data_ptr()!=model.lm_head.weight.data_ptr():
        raise AssertionError("Tied embedding/head lost in "+arm)
    if any(t.is_meta for t in model.parameters()):
        raise AssertionError("Model was not materialized")
    return model,{"changed_2D_tensors":converted_count,"unpaid_oracle":True,
                  "dense_original_weights_resident":True}


def checkpoint_modules(model):
    modules={"embedding":model.model.embed_tokens}
    if len(model.model.layers)!=30:
        raise AssertionError("Source Llama layer count drift")
    # Finer-grained first-layer probes identify a local operator error.
    for layer_id in (0,1):
        lay=model.model.layers[layer_id]
        for name in ("input_layernorm","post_attention_layernorm"):
            modules[f"layer_{layer_id:02d}.{name}"]=getattr(lay,name)
        for name in ("q_proj","k_proj","v_proj","o_proj"):
            modules[f"layer_{layer_id:02d}.attention.{name}"]=getattr(lay.self_attn,name)
        for name in ("gate_proj","up_proj","down_proj"):
            modules[f"layer_{layer_id:02d}.mlp.{name}"]=getattr(lay.mlp,name)
    for layer_id,layer in enumerate(model.model.layers):
        modules[f"block_{layer_id:02d}"]=layer
    modules["final_norm"]=model.model.norm
    modules["next_token_logits"]=model.lm_head
    return modules


@torch.inference_mode()
def traced_forward(model, token_ids):
    points={}
    handles=[]
    for name,module in checkpoint_modules(model).items():
        def capture(_mod,_inputs,output,checkpoint=name):
            tensor=output[0] if isinstance(output,(tuple,list)) else output
            if tensor.ndim!=3 or tensor.shape[0]!=1:
                raise ValueError("Wrong hidden-state tensor at "+checkpoint)
            # Freeze only one vector, never store whole sequence/logit grids.
            vector=tensor[0,-1].detach().to(torch.float32).cpu().numpy().copy()
            if not np.isfinite(vector).all():
                raise ValueError("Nonfinite model checkpoint at "+checkpoint)
            points[checkpoint]=vector
        handles.append(module.register_forward_hook(capture))
    try:
        output=model(input_ids=token_ids,use_cache=False)
        prediction=output.logits[0,-1].detach().to(torch.float32).cpu().numpy()
    finally:
        for hook in handles:hook.remove()
    if (not np.array_equal(prediction,points["next_token_logits"]) or
            len(points)!=len(checkpoint_modules(model))):
        raise AssertionError("Missing or modified pretrained checkpoint")
    return points


def distance(a,b):
    a=np.asarray(a,dtype=np.float64)
    b=np.asarray(b,dtype=np.float64)
    if a.shape!=b.shape or a.size==0:
        raise ValueError("Mismatched activation sizes")
    da=float(np.linalg.norm(a))
    db=float(np.linalg.norm(b))
    err=float(np.linalg.norm(a-b))
    return {"relative_L2_to_reference":err/max(da,1.e-25),
            "cosine":float(np.dot(a,b)/max(da*db,1.e-25)),
            "max_abs":float(np.max(np.abs(a-b))),
            "reference_l2":da,
            "candidate_l2":db,
            "dimensions":int(a.size)}


def ordered_checkpoints():
    keys=["embedding"]
    # Add all decoder boundary checkpoints, including every early layer.
    keys.extend(f"block_{i:02d}" for i in range(30))
    keys.extend(["final_norm","next_token_logits"])
    return keys


def evaluate_case(models,tok,case):
    prompt=tok.apply_chat_template([{"role":"user","content":case["prompt"]}],
        tokenize=False,add_generation_prompt=True)
    ids=torch.tensor([tok.encode(prompt,add_special_tokens=False)],dtype=torch.long)
    if ids.shape[0]!=1 or not 1<ids.shape[1]<256:
        raise ValueError("Frozen developmental prompt token count differs")
    captures={name:traced_forward(model,ids) for name,model in models.items()}
    base=captures["FP32"]
    exact=captures["V1_DENSE"]
    to_ref={name:{k:distance(base[k],candidate[k]) for k in base}
            for name,candidate in captures.items() if name!="FP32"}
    native_vs_dense={k:distance(exact[k],captures["V1_NATIVE"][k])
                    for k in exact}
    ordered=ordered_checkpoints()
    first={}
    for name,distances in to_ref.items():
        first[name]=next((k for k in ordered
            if distances[k]["relative_L2_to_reference"]>p.RELATIVE_ACTIVATION_WARNING),
            None)
    first_tokens={name:int(values["next_token_logits"].argmax())
                  for name,values in captures.items()}
    return {"case_id":case["question_id"],"domain":case["domain"],
        "question":case["question"],"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
        "prefill_token_count":int(ids.shape[-1]),
        "first_token_ids":first_tokens,
        "first_token_matches_fp32":{
            name:(value==first_tokens["FP32"]) for name,value in first_tokens.items()},
        "first_over_5pct_by_checkpoint":first,
        "distances_to_FP32":to_ref,"native_vs_same_decoded_V1":native_vs_dense}


def summarize_cases(cases):
    if len(cases)!=len(p.FROZEN_CASE_IDS):
        raise AssertionError("Incomplete public diagnostic")
    arms=[x for x in p.ARMS if x!="FP32"]
    out={}
    for arm in arms:
        gap=[case["first_over_5pct_by_checkpoint"][arm] for case in cases]
        err=[case["distances_to_FP32"][arm]["next_token_logits"]
             ["relative_L2_to_reference"] for case in cases]
        out[arm]={"first_crossing_each_case":gap,
            "relative_logits_error_mean":float(np.mean(err)),
            "match_FP32_first_token_count":sum(case["first_token_matches_fp32"][arm]
                                                  for case in cases)}
    deviations=[(key,case["native_vs_same_decoded_V1"][key]
           ["relative_L2_to_reference"]) for case in cases
           for key in ordered_checkpoints()]
    top=sorted(deviations,key=lambda item:item[1],reverse=True)[:8]
    out["native_vs_same_V1_decoded"]={
        "max_relative_error_at_main_checkpoints":max(x[1] for x in deviations),
        "worst_eight_checkpoints":top,
        "max_error_exceeds_precommitted_warning":top[0][1]>p.NATIVE_DENSE_SOURCE_ERROR_WARNING,
        "same_first_token_all_cases":all(case["first_token_ids"]["V1_DENSE"]==
                   case["first_token_ids"]["V1_NATIVE"] for case in cases)}
    return out


def run(source,out):
    torch.set_num_threads(2)
    torch.manual_seed(p.SEED)
    out.mkdir(parents=True,exist_ok=True)
    actual_bytes=source.read_bytes()
    if len(actual_bytes)!=p.MODEL_SOURCE_BYTES or hashlib.sha256(actual_bytes).hexdigest()!=p.MODEL_SOURCE_SHA256:
        raise ValueError("Different v181 V1 archive: refuse test")
    archive=CB181(actual_bytes)
    if archive.meta["original_unique_parameter_count"]!=p.TOTAL_UNIQUE:
        raise AssertionError("Changed unique pretrained source")
    if archive.meta["vector_length"]!=1 or archive.meta.get("calibri_recovery_oracle") is not True:
        raise AssertionError("Not the original V1 diagnostic geometry")
    selected=[c for c in freeze_cases(all_cases()) if c["question_id"] in p.FROZEN_CASE_IDS]
    if len(selected)!=4 or [c["question_id"] for c in selected]!=list(p.FROZEN_CASE_IDS):
        raise AssertionError("Frozen public diagnostic sample changed")
    settings={"model":MODEL,"revision":MODEL_SHA,"physical_V1_sha256":p.MODEL_SOURCE_SHA256,
        "physical_V1_bytes":p.MODEL_SOURCE_BYTES,"physical_V1_bpp":p.V1_TOTAL_BPP,
        "original_unique_parameter_count":p.TOTAL_UNIQUE,
        "cases":selected,"arms":p.ARMS,
        "first_warning_relative_activation":p.RELATIVE_ACTIVATION_WARNING,
        "native_vs_dense_warning":p.NATIVE_DENSE_SOURCE_ERROR_WARNING,
        "source_fit_or_training_in_this_study":False,
        "source_decoded_dense_and_interventions_NOT_DEPLOYABLE":True,
        "independent_PRIVATE_TEST_inaccessible":True,
        "Jessica_runs_no_tests":True}
    save(out/"frozen_protocol_before_inference.json",settings)
    teacher=original_teacher()
    source_audit=validate_teacher_source(teacher,archive)
    decoded={}
    original=teacher.state_dict()
    source_audit["matrix_source_checks"]={}
    for name,d in sorted(archive.meta["tensor_descriptors"].items()):
        if d["dtype"]!="CB181_K32V1":continue
        w=decode_handle(archive.handles[name])
        decoded[name]=w
        true=original[name].detach().cpu().numpy()
        if true.shape!=w.shape:raise ValueError("Shape corruption: "+name)
        if name=="model.embed_tokens.weight" or name.endswith(".down_proj.weight") and name.startswith("model.layers.0."):
            source_audit["matrix_source_checks"][name]={
                "relative_weight_error":distance(true.ravel(),w.ravel())["relative_L2_to_reference"]}
    if len(decoded)!=211:raise AssertionError("Missing trained original matrices")
    source_audit["decoded_trained_2D_count"]=len(decoded)
    save(out/"original_source_audit.json",source_audit)
    models={"FP32":teacher}
    variant_notes={}
    for arm in ("FP16_ROUND","V1_DENSE","V1_HEAD_ONLY","V1_BODY_ONLY"):
        models[arm],variant_notes[arm]=make_dense_variant(teacher,archive,arm,decoded)
    models["V1_NATIVE"],physical,_=build_native(actual_bytes,"ORACLE_V1",out/"native_kernel")
    if physical.hash()!=p.MODEL_SOURCE_SHA256:
        raise AssertionError("Runtime source SHA mismatch")
    if any(param.ndim>=2 for param in models["V1_NATIVE"].parameters()):
        raise AssertionError("Native V1 leaked original dense pretrained matrices")
    del decoded
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    model_result=[]
    for case in selected:
        current=evaluate_case(models,tokenizer,case)
        model_result.append(current)
        save(out/"checkpoint_trace_partial.json",model_result)
        print(json.dumps({"stage":"trace_case","id":case["question_id"],
            "first_crossings":current["first_over_5pct_by_checkpoint"],
            "first_tokens":current["first_token_ids"],
            "max_native_dense_error":max(v["relative_L2_to_reference"]
                 for v in current["native_vs_same_decoded_V1"].values())}),flush=True)
    summary=summarize_cases(model_result)
    report={"completed":True,"protocol":settings,"source_integrity":source_audit,
        "variants":variant_notes,"trace":model_result,"summary":summary,
        "not_a_0p7_candidate":True,"not_real_answer_quality_test":True,
        "safe_only_research":True}
    save(out/"v182_complete.json",report)
    print(json.dumps({"stage":"complete","summary":summary}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    run(args.source,args.output)
