"""v101: distinguish native head codec defects from unavoidable source loss.

Exactly original pretrained FP32 teacher and byte-for-byte frozen CB59
compressed runtime. Same-input prefill; all 49152 output rows decoded in
temporary scratch blocks for a read-only oracle, not a deployed dense model.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
import torch
from transformers import AutoTokenizer
from direct_codebook_v59 import CB59
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v98.metrics import logits_topk
from v99.patch import capture_one
from .codec import exact_streamed_audit
from .spec import PROBES,EXPLICIT_ROW_INDICES

def capture_norm_and_logits(model,ids):
    with torch.inference_mode(),capture_one(model,"model.norm") as observed:
        logits=model(input_ids=ids,use_cache=False).logits[0,-1,:].detach().clone()
    if len(observed)!=1:raise AssertionError("last pretrained norm output not captured")
    hidden=observed[0][0,-1,:].float().clone()
    return hidden,logits.float()

def topcompare(reference,actual,tok):
    x=logits_topk(reference,tok)
    y=logits_topk(actual,tok)
    return {"reference_top10_overlap":len(set(x["top10_token_ids"])&set(y["top10_token_ids"])),
        "reference_top1_matched":x["top10_token_ids"][0]==y["top10_token_ids"][0],
        "reference_top1_text":x["top10_text"][0],
        "observed_top1_text":y["top10_text"][0],
        "reference_top1_probability":x["top1_probability"],
        "observed_top1_probability":y["top1_probability"],
        "teacher_logits_distribution_not_semantic_quality":True}

def run(original:Path,locked:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    source=original.read_bytes()
    frozen=json.loads(locked.read_text())
    physical=frozen["frozen_reference_sources"]["CB59"]
    digest=hashlib.sha256(source).hexdigest()
    bpw=8*len(source)/EXPECTED_UNIQUE
    if (frozen.get("contract_version")!="CALIBRI_BASELINE_LOCK_2026-11-04" or
        digest!=physical["sha256"] or len(source)!=physical["complete_archive_bytes"] or
        bpw>.700 or MODEL_SHA!=frozen["original_model_revision"]):
        raise AssertionError("pinned original physical source SHA or rate changed")
    source_archive=CB59(source)
    alias=source_archive.handles["model.embed_tokens.weight"] is source_archive.handles["lm_head.weight"]
    if not alias:raise AssertionError("tied source head and embedding codebook got duplicated")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    student,info,_=make_model("baseline",SimpleNamespace(original=original,folder=out))
    if not info["zero_dense_pretrained_2D_hot_weights"]:raise AssertionError("packed native source missing")
    teacher=original_teacher()
    if (student.lm_head.handle is not student.get_input_embeddings().handle or
        teacher.get_input_embeddings().weight.data_ptr()!=teacher.lm_head.weight.data_ptr()):
        raise AssertionError("original trained tied output/head alias mismatch")
    pairs=[]
    for domain,prompt in PROBES:
        encoded=tok.encode(prompt,add_special_tokens=False)
        if not (1<len(encoded)<=96):raise AssertionError("invalid original new prompt")
        input_ids=torch.tensor(encoded,dtype=torch.long)[None,:]
        teacher_hidden,teacher_full=capture_norm_and_logits(teacher,input_ids)
        native_hidden,native_full=capture_norm_and_logits(student,input_ids)
        pairs.append({"domain":domain,
            "prompt_SHA256":hashlib.sha256(prompt.encode()).hexdigest(),
            "teacher_hidden":teacher_hidden,
            "native_hidden":native_hidden,
            "teacher_logits":teacher_full,
            "native_logits":native_full})
    all_h=torch.stack([row[key] for row in pairs
                for key in ("teacher_hidden","native_hidden")])
    with torch.inference_mode():
        native_head_all=student.lm_head(all_h).detach().cpu().numpy()
        exact_head_all=teacher.lm_head(all_h).detach().cpu().numpy()
    # Native final model logits must be EXACT same as its own frozen packed head.
    for i,entry in enumerate(pairs):
        if not np.allclose(native_head_all[i*2+1],entry["native_logits"].numpy(),rtol=1e-5,atol=1e-5):
            raise AssertionError("native head direct versus full model logits not identical")
        if not np.allclose(exact_head_all[i*2],entry["teacher_logits"].numpy(),rtol=1e-5,atol=1e-5):
            raise AssertionError("pinned exact teacher head versus full-model logits differ")
    audit=exact_streamed_audit(
        student.lm_head.handle,teacher.lm_head.weight,all_h.numpy(),native_head_all)
    samples=audit.pop("selected_full_decoded_rows")
    _=audit.pop("decoded_logits")
    actual_ids=[k for k in EXPLICIT_ROW_INDICES if k in samples]
    ids=torch.tensor(actual_ids,dtype=torch.long)
    native_emb=student.get_input_embeddings()(ids).detach().numpy()
    emb_max=float(max(np.max(np.abs(native_emb[k]-samples[row]))
                      for k,row in enumerate(actual_ids)))
    if emb_max>1e-6:
        raise AssertionError("C demand embedding differs from Python reconstructed original head row")
    audit["native_C_tied_embedding_vs_explicit_Python_head_row_max_abs_diff"]=emb_max
    audit["native_tied_embedding_source_same_as_output_head"]=True
    breakdown=[]
    for i,row in enumerate(pairs):
        ref=row["teacher_logits"]
        # Unmodified original compact model
        baseline=row["native_logits"]
        # Only codec loss, identical exact original teacher hidden.
        quant_only=torch.from_numpy(native_head_all[2*i].copy())
        # Only distributed hidden activation corruption, exact output head.
        hidden_only=torch.from_numpy(exact_head_all[2*i+1].copy())
        # Independent compact model head-native logit output (both corruptions).
        if not np.allclose(native_head_all[2*i+1],baseline.numpy(),atol=1e-5,rtol=1e-5):
            raise AssertionError("unpaired native full-logit snapshot")
        metrics={"original_FP32_vs_native_both_faults":topcompare(ref,baseline,tok),
                 "original_FP32_hidden_with_exact_native_codebook":topcompare(ref,quant_only,tok),
                 "native_corrupt_hidden_with_original_FP32_head":topcompare(ref,hidden_only,tok)}
        breakdown.append({"domain":row["domain"],
           "original_prompt_SHA256":row["prompt_SHA256"],
           "unmodified_original_native_logit_predictions":metrics,
           "teacher_vs_native_hidden_relative_RMS":float(
               torch.linalg.vector_norm(row["teacher_hidden"]-row["native_hidden"])/
               torch.linalg.vector_norm(row["teacher_hidden"]).clamp_min(1e-10))})
        print(json.dumps({"stage":"v101_true_original_head_encoding_loss_on_same_activations",
            "domain":row["domain"],
            "top10_overlap_native":metrics["original_FP32_vs_native_both_faults"]["reference_top10_overlap"],
            "top10_overlap_codec_only":metrics["original_FP32_hidden_with_exact_native_codebook"]["reference_top10_overlap"],
            "top10_overlap_hidden_only":metrics["native_corrupt_hidden_with_original_FP32_head"]["reference_top10_overlap"],
            "native_vs_python_codec_max_absolute_logit_diff":
                audit["native_C_vs_explicit_Python_decoder_max_abs_logit_diff"]},
            ensure_ascii=False),flush=True)
    report={"experiment":"CALIBRI_v101_original_head_encoder_decoder_loss_vs_native_C_parity",
        "unchanged_original_CB59_SHA256":digest,
        "unchanged_original_CB59_complete_bytes":len(source),
        "all_in_bpp":bpw,
        "model_revision":MODEL_SHA,
        "full_original_tied_input_output_5bit_source":True,
        "raw_5bit_index_roundtrip_and_codec_weight_metrics":audit,
        "new_synthetic_same_prefix_samples":breakdown,
        "did_not_modify_or_save_any_trained_weights":True,
        "full_dense_decoder_is_diagnostic_scratch_only_NEVER_runtime":True,
        "exact_original_pretrained_FP32_teacher_head_unpaid_reference":True,
        "official_independent_v89_TEST_unopened":True,
        "no_live_Jessica_main_Render_Supabase_APK_changes":True,
        "interpretation":"Native C and Python codec equivalence checks for software decoding "
          "and packing errors, while pretrained FP32 source difference measures lossy "
          "encoder information loss. No quality gain or deployable repair is claimed."}
    outfile=out/"v101_head_codec_exact_roundtrip_vs_pretrained_loss.json"
    outfile.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"stage":"v101_completed_codebook_codec_and_original_weight_quality_audit",
        "weight_relative_RMSE":audit["original_FP32_to_CB59_head_weight_relative_RMSE"],
        "C_python_max_abs_logit_diff":audit["native_C_vs_explicit_Python_decoder_max_abs_logit_diff"],
        "source_bytes":len(source),
        "official_TEST_untouched":True}),flush=True)
    return report
