"""True original pretrained weights -> physically paid 8-bank tied head.

Fully isolates candidate from old official v89 TEST and Live. Full
source archive and original base are SHA verified before source recoding.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from paid_lowrank_v76 import make_model
from direct_codebook_v59 import CB59
from v84.sources import original_teacher
from v94.generation import generate_one
from v96.grade import grade_answer,string_breakdown
from v98.metrics import logits_topk
from .spec import PROBES,VOCAB,ROW_PART
from .physical import unpack,result_audit
from .encode import recode_original
from .runtime import bind_native_banks
from .metrics import decode_rows,full_head_evidence

def compare(ref,other,tok):
    a=logits_topk(ref,tok);b=logits_topk(other,tok)
    return {"teacher_top10_overlap":len(set(a["top10_token_ids"])&
                                       set(b["top10_token_ids"])),
        "teacher_top1_matched":a["top10_token_ids"][0]==b["top10_token_ids"][0],
        "teacher_top1":a["top10_text"][0],"native_top1":b["top10_text"][0],
        "not_ground_truth_or_semantic_quality":True}

def generate(model,tok,prompt):
    output=generate_one(model,tok,prompt)
    return {"actual_text":output["text"],"generated_ids":output["ids"],
        "visible_lexical":string_breakdown(output["text"],output["ids"]),
        "actual_native_CPU_KV_cache_full_token_steps":output["full_model_token_steps"],
        "elapsed_seconds":output["elapsed_seconds"],
        "semantics_not_proven_by_distinct_tokens":True}

def run(source_path:Path,lock_path:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    frozen=source_path.read_bytes()
    lock=json.loads(lock_path.read_text())
    origin=lock["frozen_reference_sources"]["CB59"]
    frozen_sha=hashlib.sha256(frozen).hexdigest()
    if (lock.get("contract_version")!="CALIBRI_BASELINE_LOCK_2026-11-04"
        or frozen_sha!=origin["sha256"] or
        len(frozen)!=origin["complete_archive_bytes"] or
        MODEL_SHA!=lock["original_model_revision"]):
        raise AssertionError("not original full-source frozen pretrained model")
    teacher=original_teacher()
    with torch.inference_mode():
        candidate_bytes,fit=recode_original(frozen,teacher.lm_head.weight)
    whole=out/"CALIBRI_v104_eight_bank_native_complete_source.cb104"
    whole.write_bytes(candidate_bytes)
    embedded,banks=unpack(candidate_bytes)
    nested=out/"temporary_embedded_original_layout_for_CB59_initialization.cb59"
    nested.write_bytes(embedded)
    cb=CB59(embedded)
    if cb.handles["lm_head.weight"] is not cb.handles["model.embed_tokens.weight"]:
        raise AssertionError("head tied to embedding was duplicated")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    root_original=out/"cb59_reference";root_original.mkdir()
    root_v104=out/"v104_native";root_v104.mkdir()
    original,old_info,_=make_model("baseline",SimpleNamespace(
        original=source_path,folder=root_original))
    transient,info,_=make_model("baseline",SimpleNamespace(
        original=nested,folder=root_v104))
    model,banked_info=bind_native_banks(transient,banks,Path(__file__).with_name("native104.c"),
                                      root_v104/"libCB104_native.so")
    if not old_info["zero_dense_pretrained_2D_hot_weights"] or not info["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("nonpacked original pretrained 2D source leaked")
    diag=[]
    activations=[]
    for domain,prompt in PROBES:
        ids=torch.tensor(tok.encode(prompt,add_special_tokens=False),dtype=torch.long)[None,:]
        if not 1<ids.numel()<=96:raise AssertionError("wrong new no-TEST prompt")
        with torch.inference_mode():
            fp=teacher(input_ids=ids,use_cache=False).logits[0,-1].detach().float()
            old=original(input_ids=ids,use_cache=False).logits[0,-1].detach().float()
            new=model(input_ids=ids,use_cache=False).logits[0,-1].detach().float()
        record={"domain":domain,
            "prompt_SHA256":hashlib.sha256(prompt.encode()).hexdigest(),
            "original_CB59":compare(fp,old,tok),
            "banked_native_CB104":compare(fp,new,tok)}
        diag.append(record)
        print(json.dumps({"stage":"v104_same_prefix_actual_full_native_C_head",
          "domain":domain,
          "original_top10":record["original_CB59"]["teacher_top10_overlap"],
          "new_banked_top10":record["banked_native_CB104"]["teacher_top10_overlap"]}),
          flush=True)
    sample_heads=[]
    # Compare EXACT same genuine input to native C and temporary Python
    # reference head reconstruction. No teacher activation used by model.
    for domain,prompt in PROBES[:2]:
        ids=torch.tensor(tok.encode(prompt,add_special_tokens=False),dtype=torch.long)[None,:]
        captured=[]
        def take(_module,arguments):
            captured.append(arguments[0].detach()[0,-1].clone())
        handle=model.lm_head.register_forward_pre_hook(take)
        try:
            with torch.inference_mode():_ = model(input_ids=ids,use_cache=False)
        finally:handle.remove()
        if len(captured)!=1:raise AssertionError("missing native original head activation")
        sample_heads.append(captured[0].float().numpy())
    source_vec=np.stack(sample_heads)
    native_output=model.lm_head(torch.from_numpy(source_vec)[:,None,:])[:,0,:].detach().numpy()
    head_fidelity=full_head_evidence(model.lm_head.handle,banks,teacher.lm_head.weight,
                                    source_vec,native_output)
    token_ids=[0,6143,6144,12287,12288,49151]
    with torch.inference_mode():
        native_emb=model.get_input_embeddings()(torch.tensor(token_ids)).detach().numpy()
    original_rows=np.concatenate([decode_rows(model.lm_head.handle,banks,k,k+1)
                                 for k in token_ids],axis=0)
    embed_diff=float(np.max(np.abs(native_emb-original_rows)))
    if embed_diff>1e-6:raise AssertionError("CB104 C input and output codebook do not alias")
    head_fidelity["C_embedding_vs_same_tied_Python_head_rows_max_abs"]=embed_diff
    # Small paired true generation on native model, standard CPU KV cache.
    chat_cases=(
      ("math","What is 5 times 9? Reply with only the number.","45"),
      ("geography","What is the capital of Norway? Reply with only the city.","Oslo"),
    )
    text_reports=[]
    for i,(domain,question,answer) in enumerate(chat_cases):
        prompt=tok.apply_chat_template([{"role":"user","content":question}],
                            tokenize=False,add_generation_prompt=True)
        original_models=(("CB59",original),("CB104_banked",model))
        if i%2:original_models=tuple(reversed(original_models))
        scored={name:generate(m,tok,prompt) for name,m in original_models}
        scored["FP32"]=generate(teacher,tok,prompt)
        for value in scored.values():
            value["conservative_first_answer_match"]=grade_answer(value["actual_text"],[answer])["correct"]
        text_reports.append({"domain":domain,"gold":answer,
            "chat_prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
            "original_chat_template_KV_cached_full_model_texts":scored})
        print(json.dumps({"stage":"v104_original_chats_real_gold",
          "domain":domain,"gold":answer,
          "CB59_correct":scored["CB59"]["conservative_first_answer_match"],
          "CB104_correct":scored["CB104_banked"]["conservative_first_answer_match"],
          "FP32_correct":scored["FP32"]["conservative_first_answer_match"]}),flush=True)
    def speed(kind):
        return sum(x["original_chat_template_KV_cached_full_model_texts"][kind]["actual_native_CPU_KV_cache_full_token_steps"]
                   for x in text_reports)/max(1e-10,sum(
            x["original_chat_template_KV_cached_full_model_texts"][kind]["elapsed_seconds"]
            for x in text_reports))
    speed_reference=speed("CB59")
    speed_banked=speed("CB104_banked")
    outdata={
        "experiment":"CALIBRI_v104_physically_paid_native_eight_bank_whole_tied_head",
        "source_original_FROZEN_SHA":frozen_sha,
        "entire_new_archive_sha256":hashlib.sha256(candidate_bytes).hexdigest(),
        "physical_archive_with_extra_banks":result_audit(candidate_bytes),
        "source_only_original_weight_fit":fit,
        "original_fp32_head_vs_native_banked_source":head_fidelity,
        "original_model_revision":MODEL_SHA,
        "new_synthetic_same_prefix_top10_NOT_TEST":diag,
        "real_CPU_KV_cache_32_token_original_chat_template":text_reports,
        "true_baseline_native_tokens_per_sec":speed_reference,
        "true_eight_bank_native_tokens_per_sec":speed_banked,
        "native_8bank_vs_CB59_CPU_speed_ratio":speed_banked/speed_reference,
        "speed_at_least_90pct":speed_banked/speed_reference>=.90,
        "new_8bank_native_model_selected_or_deployed":False,
        "original_weights_and_CB59_archive_unchanged":True,
        "teacher_F32_not_used_by_live_CB104_inference":True,
        "no_official_independent_TEST_opened":True,
        "no_Jessica_main_Render_Supabase_APK_modification":True,
        "scientific_semantic_quality_unverified":True,
        "interpretation":"Whole source 8bank recoding is physical/native but lexical diversity "
            "or source RMSE alone cannot certify usable task performance. Preserve "
            "generated exact text and compare to independent gold."}
    save=out/"v104_actual_original_head_bank_source_real_CPU_and_semantic_tests.json"
    save.write_text(json.dumps(outdata,indent=2,ensure_ascii=False)+"\n")
    # Temporary exploded-source artifact is not needed; ensure single CB104
    # physical file is the only new model artifact uploaded.
    nested.unlink()
    print(json.dumps({"stage":"v104_finished_physically_paid_source_banked_C",
        "source_RMSE":head_fidelity["genuine_pretrained_original_head_weight_relative_RMSE"],
        "all_in_bpp":8*len(candidate_bytes)/EXPECTED_UNIQUE,
        "speed_ratio":speed_banked/speed_reference,
        "original_TEST_untouched":True}),flush=True)
    return outdata
