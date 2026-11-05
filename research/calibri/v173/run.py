"""Real complete-model C-vs-dense-source round-trip forensic audit.

The unpaid diagnostic materializes the SAME low-bit CB59 weight values
in FP32 PyTorch, NOT original teacher weights. If native CB59 and
that reconstruction both output the same nonsense, there is no
plausible native bit-unpacker regression that would fix semantic quality.

No changing CODEBOOK, no NLL, KL, teacher distribution fitting or QA labels.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer
from direct_codebook_v59 import CB59
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE
from v153.codec import unpack,report
from v162.evidence import frozen_v161
from v108.generation import build_chat
from v94.generation import NEW_PROMPTS,generate_one
from binary_geometry_v64 import MODEL,MODEL_SHA

from .protocol import (ARMS,PREFILL_CASE_IDS,GENERATION_CASE_IDS,
                       EXPECTED_BYTE_SIZE,check)
from .recover import build_unpaid_reference,exact_dense_matrix


@torch.inference_mode()
def matrix_roundtrip(archive,native):
    samples=(
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.10.mlp.up_proj.weight",
        "model.layers.29.mlp.down_proj.weight",
    )
    out={}
    for i,name in enumerate(samples):
        h=archive.handles[name]
        raw=exact_dense_matrix(h)
        rng=np.random.default_rng(173+i)
        x=(rng.normal(size=(2,h.shape[1]))/np.sqrt(h.shape[1])).astype(np.float32)
        c=native.kernel.linear(h,x)
        dense=x@raw.T
        abs_diff=float(np.max(np.abs(c-dense)))
        rms=float(np.sqrt(np.mean((c-dense).astype(np.float64)**2)))
        rel=float(np.linalg.norm((c-dense).astype(np.float64))/
                  max(np.linalg.norm(dense.astype(np.float64)),1e-15))
        if not np.isfinite(c).all() or not np.isfinite(dense).all():
            raise ValueError("nonfinite native C or reconstructed compressed source")
        out[name]={"shape":list(h.shape),"maximum_absolute_C_vs_dense":abs_diff,
                   "rms_C_vs_dense":rms,"relative_Frobenius_C_vs_dense":rel}
        if rel>3e-4:
            raise AssertionError(f"candidate real native CB59 kernel mismatch at {name}: {rel}")
    # Embedding/LM tied source: same codes, embedding on-demand C vs
    # exact literal matrix storage in diagnostic.
    h=archive.handles["model.embed_tokens.weight"]
    ids=np.array([0,1,2,19,128,329,2000,49151],dtype=np.int64)
    real=native.kernel.embedding(h,ids)
    dense=exact_dense_matrix(h)[ids]
    out["tied_embedding_native_vs_dense"]={
        "max_absolute":float(np.max(np.abs(real-dense))),
        "tied_alias_identity":archive.handles["lm_head.weight"] is h,
    }
    if not np.allclose(real,dense,rtol=0,atol=1e-7):
        raise AssertionError("native C embedding and dense equivalent disagree")
    return out


@torch.inference_mode()
def matched_prefill(models,tok):
    result=[]
    for idx in PREFILL_CASE_IDS:
        category,question=NEW_PROMPTS[idx]
        prompt=build_chat(tok,question)
        ids=torch.tensor([tok.encode(prompt,add_special_tokens=False)],dtype=torch.long)
        logits={}
        for arm in ARMS:
            x=models[arm](input_ids=ids,use_cache=False).logits[0,-1].detach()
            if x.shape!=(49152,) or not torch.isfinite(x).all():
                raise ValueError("full-model logits wrong shape or NaN")
            logits[arm]=x
        source=logits[ARMS[1]]
        dense=logits[ARMS[2]]
        diff=(source-dense).abs()
        result.append({
            "id":idx,"domain":category,"prompt":question,
            "native_vs_dense_reconstructed_max_abs":float(diff.max()),
            "native_vs_dense_reconstructed_mean_abs":float(diff.mean()),
            "native_and_dense_first_argmax_agree":bool(source.argmax()==dense.argmax()),
            "native_first_token_id":int(source.argmax()),
            "dense_same_source_first_token_id":int(dense.argmax()),
            "original_FP32_teacher_first_token_id":int(logits[ARMS[0]].argmax()),
            "both_compact_arms_not_original_FP32_source":True,
        })
    return result


@torch.inference_mode()
def free_text(models,tok):
    result={a:[] for a in ARMS}
    for idx in GENERATION_CASE_IDS:
        cat,question=NEW_PROMPTS[idx]
        prompt=build_chat(tok,question)
        for arm in ARMS:
            out=generate_one(models[arm],tok,prompt,stop_at_eos=True)
            tokens=out["ids"]
            result[arm].append({
                "id":idx,"category":cat,"question":question,
                "actual_generated_text":out["text"],
                "actual_generated_token_ids":tokens,
                "actual_EOS":out["eos_generated"],
                "actual_num_tokens":len(tokens),
                "distinct_ratio":len(set(tokens))/len(tokens),
                "native_full_model_wall_seconds":out["elapsed_seconds"],
            })
    same=[{
        "id":idx,
        "same_generated_token_sequence":a["actual_generated_token_ids"]==b["actual_generated_token_ids"],
        "same_verbatim_response":a["actual_generated_text"]==b["actual_generated_text"],
    } for idx,a,b in zip(GENERATION_CASE_IDS,result[ARMS[1]],result[ARMS[2]])]
    return result,same


def run(physical,old_report,seal,out):
    check()
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    raw=physical.read_bytes()
    paid=unpack(raw)
    cb59=paid.original
    if len(cb59)!=EXPECTED_BYTE_SIZE or hashlib.sha256(cb59).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("not original real approved CB59 pretrained complete source")
    provenance=frozen_v161(paid,old_report,seal)
    protocol={
        "stage":"CALIBRI_v173_REAL_NATIVE_VS_IDENTICAL_DEQUANT_SOURCE",
        "actual_original_CB59_bytes":len(cb59),
        "actually_paid_CB161_bytes":len(raw),
        "original_unique_parameters":TOTAL_UNIQUE,
        "original_CB59_bpp":8*len(cb59)/TOTAL_UNIQUE,
        "existing_CB161_bpp":8*len(raw)/TOTAL_UNIQUE,
        "complete_original_SHA":ORIGINAL_CB59_SHA,
        "precommitted_prefill_ids":list(PREFILL_CASE_IDS),
        "precommitted_full_generation_ids":list(GENERATION_CASE_IDS),
        "NO_KL_NLL_PPL_as_quality_criterion":True,
        "FULL_FP32_FROM_COMPRESSED_MATRIX_IS_DIAGNOSTIC_NOT_DEPLOYABLE":True,
        "no_v89_independent_TEST_opened":True,
        "no_model_weights_changed":True,
        "no_LIVE_Jessica_main_Render_Supabase_APK_change":True,
    }
    (out/"v173_SOURCE_AND_PROBES_FROZEN_BEFORE_MODELS.json").write_text(
        json.dumps(protocol,ensure_ascii=False,indent=2)+"\n")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    teacher=original_teacher()
    with TemporaryDirectory(dir=out) as tmp:
        root=Path(tmp)
        (root/"native").mkdir()
        src=root/"frozen_original_CB59.cb59"
        src.write_bytes(cb59)
        native,meta,_=make_model("baseline",SimpleNamespace(original=src,folder=root/"native"))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise AssertionError("actual native model loaded dense original weights")
        if any(p.ndim>=2 for p in native.parameters()):
            raise AssertionError("real physically compressed inference has dense trained matrix")
        disk=CB59(cb59)
        source_kernel=matrix_roundtrip(disk,native.lm_head)
        diagnostic,diagnostic_meta=build_unpaid_reference(cb59)
        models={ARMS[0]:teacher,ARMS[1]:native,ARMS[2]:diagnostic}
        prefill=matched_prefill(models,tok)
        original_text,same=free_text(models,tok)
        if any(p.ndim>=2 for p in native.parameters()):
            raise AssertionError("native source expanded after inference")
    raw_result={
        "stage":protocol["stage"],"sealed":protocol,"source_v161":provenance,
        "original_CB59_real_source_and_C_kernel_vs_dense_comparisons":source_kernel,
        "UNPAID_dense_reference_explicitly_not_an_executable_compressed_model":diagnostic_meta,
        "actual_original_FP32_vs_native_CB59_vs_exact_decompressed_CB59_first_token":prefill,
        "all_actual_native_and_dense_and_original_teacher_greedy_responses":original_text,
        "native_identical_compressed_dense_source_greedy_agreement":same,
        "full_C_v59_decoder_kernel_matrices_211_already_materialized_in_reference":True,
        "unresolved_if_nonmatching_do_not_assume_source_quantization_is_sole_cause":True,
        "no_answer_improvement_or_new_codec_promised":True,
        "no_unpaid_additional_weight_or_hidden_teacher_oracle_used_in_native_model":True,
    }
    path=out/"v173_native_vs_exact_dense_CB59_reference_full_answers.json"
    path.write_text(json.dumps(raw_result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":protocol["stage"],
        "C_matrix_kernel_roundtrip":source_kernel,
        "native_vs_IDENTICAL_dequantized_CB59_logits_agreement":
            sum(x["native_and_dense_first_argmax_agree"] for x in prefill),
        "native_vs_identical_dequantized_CB59_generated_text_agreement":
            sum(x["same_verbatim_response"] for x in same),
        "out_of_prefill_probes":len(prefill),
        "out_of_full_generation_probes":len(same),
        "report":str(path)
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("original_paid_CB161",type=Path)
    a.add_argument("v161_completed_report",type=Path)
    a.add_argument("v161_before_seal",type=Path)
    a.add_argument("out",type=Path)
    args=a.parse_args()
    run(args.original_paid_CB161,args.v161_completed_report,args.v161_before_seal,args.out)
