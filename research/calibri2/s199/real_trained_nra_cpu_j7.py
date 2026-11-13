"""S199: independent post-S198 *trained* NRA physically loaded 616 CPU J7.

The S198 real four-NEW-TRAIN backward step changed 10,240 FP16 decoder
coefficients while preserving all original S197 tensors. S199 repackages its
candidate with lossless LZMA to recover physical <=0.7 bpp headroom and runs
the unchanged original S190→S187→S168 CPU inference on exactly 456+128+32.

All tokens are captured before comparison with original TRAIN labels.
Fail-closed if forgetting any protected original full/EOS or first token,
all-inclusive physical bpp>0.7, or true sample ratio<.9. Only gain ≥1/32
is called a useful improvement. No QA/TEST and no LIVE.
"""
from __future__ import annotations
import hashlib
import json
import lzma
from pathlib import Path
import sys

import torch
from safetensors.torch import load
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from calibri2.s132.progressive_core import IDS_KEY, WEIGHTS_KEY
from calibri2.s190.real_cpu_new_train_paraphrase_probe import new_train_paraphrase_probes
from calibri2.s195_neuro_area.representation_area import NeuroRepresentationArea,PREFIX as NRA_PREFIX
from calibri2.s194.actual_cpu_j7_bridge import (
    make_real_584_certificate, sha_text,
)
from calibri2.s197.real_46m_nra_j7 import new_46m_core_type
from calibri2.s196.core_growth_budget import NEXT_TOTAL_PARAMS,PREVIOUS_PARAMS

S198_PROOF=43370918910
S197_PINNED_BPP=43369398297
S199_STAGE="S199_INDEPENDENT_REAL_TRAINED_NRA_46M_616_CPU_J7"
MIN_CPU_RATIO=.9
MAX_BPP=.7


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def physical_and_parent(parent_zlib:Path,parent_report:Path,
                        candidate_zlib:Path,train_report:Path,
                        destination:Path):
    import zlib
    from safetensors.torch import save
    if destination.exists():
        raise FileExistsError("S199 cannot overwrite a candidate runtime package")
    p=json.loads(parent_report.read_text())
    tr=json.loads(train_report.read_text())
    base=parent_zlib.read_bytes()
    learned=candidate_zlib.read_bytes()
    if (p.get("stage")!="S197_ACTUAL_PHYSICAL_46M_NRA_REAL_CPU_616_J7"
        or p.get("exact_protected_584_generated_ids_equal_S194") is not True
        or p.get("all_in_physical_bpp",{}).get(
            "conservative_excluding_new_zero_growth_bpp",1)>.7
        or p.get("checkpoint_runtime_physical",{}).get("runtime_packed_sha256")!=sha_bytes(base)
        or tr.get("stage")!="S198_GENUINE_ONE_CPU_NRA_GRADIENT_STEP_TRAIN_ONLY"
        or tr.get("actual_CPU_train_steps")!=1
        or tr.get("model_accepted") is not False
        or tr.get("all_legacy_non_NRA_tensors_byte_identical") is not True
        or tr.get("parent_package_SHA256")!=sha_bytes(base)
        or tr.get("candidate_package_SHA256")!=sha_bytes(learned)
        or tr.get("train_NRA_Fp16_changed_elements",{}).get(
            NRA_PREFIX+"decoder.weight",0)<1):
        raise AssertionError("S199 independent physical S197/S198 provenance invalid")
    original=load(zlib.decompress(base))
    updated=load(zlib.decompress(learned))
    if (set(original)!=set(updated)
        or updated[WEIGHTS_KEY].shape!=(12,320,72)
        or any(not torch.equal(original[k],updated[k]) for k in original
               if not k.startswith(NRA_PREFIX))
        or not any(not torch.equal(original[k],updated[k]) for k in original
                   if k.startswith(NRA_PREFIX))):
        raise AssertionError("S199 S198 candidate changed old 46M, GT, or no neural weight")
    # Format is new, self-contained and lossless. TRAIN artifact zlib is NOT
    # a deployed second copy. Runtime physically persists only this LZMA file.
    physical_blob=save(updated)
    compressed=lzma.compress(physical_blob,preset=6)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_bytes(compressed)
    recovered=load(lzma.decompress(destination.read_bytes()))
    if (set(recovered)!=set(updated)
        or any(not torch.equal(recovered[k],v) for k,v in updated.items())):
        raise AssertionError("S199 physical trained NRA LZMA/FP16 roundtrip failed")
    if not torch.equal(recovered[WEIGHTS_KEY],original[WEIGHTS_KEY]):
        raise AssertionError("S199 original rank1344 46.7M main weights mutated")
    return p,tr,recovered,{
        "parent_package_bytes":len(base),
        "candidate_intermediate_zip_bytes":len(learned),
        "actual_single_runtime_lzma_bytes":len(compressed),
        "actual_runtime_lzma_SHA256":sha_bytes(compressed),
        "raw_all_FP16_bytes":len(physical_blob),
        "actual_runtime_FP16_sha256":sha_bytes(physical_blob),
        "new_NRA_FP16_train_changed":{
            k:int(torch.count_nonzero(original[k]!=recovered[k]))
            for k in updated if k.startswith(NRA_PREFIX)},
    }


def physical_invoice(*,parent_report,physical,manifest_bytes,source_paths):
    parent_bits=float(parent_report["all_in_physical_bpp"]["all_in_paid_bits"])
    prior=int(physical["parent_package_bytes"])
    packed=int(physical["actual_single_runtime_lzma_bytes"])
    source_bytes=sum(Path(p).stat().st_size for p in source_paths)
    if parent_bits<=0 or prior<=0 or packed<=0 or manifest_bytes<0:
        raise AssertionError("S199 physical all-inclusive invoice invalid")
    bits=parent_bits -8*prior +8*(packed+source_bytes+manifest_bytes)
    return {
        "all_in_bits":bits,
        "architectural_bpp":bits/NEXT_TOTAL_PARAMS,
        "conservative_bpp_excluding_zero_structural_growth":bits/(PREVIOUS_PARAMS+21_900),
        "replaced_S197_package_bytes":prior,
        "persisted_lzma_runtime_bytes":packed,
        "S198_S199_runtime_source_bytes_charged":source_bytes,
        "this_manifest_bytes_charged":manifest_bytes,
    }


def main():
    if len(sys.argv)<15:
        raise SystemExit("S199 exact S197 native 616 CPU CLI required")
    args=list(sys.argv)
    out=Path(args[-1])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S199 requires fresh separate output")
    out.mkdir(parents=True,exist_ok=True)
    parent_z=Path("inputs/s197/S197_S142_DERIVED_46M_PLUS_NRA_ALL_FP16.safetensors.zlib")
    parent_j7=Path("inputs/s197/S197_REAL_46M_NRA_616_CPU_J7.json")
    cand_z=Path("inputs/s198/S198_UNACCEPTED_ONE_TRUE_CPU_NRA_STEP_46M_FP16.safetensors.zlib")
    cand_train=Path("inputs/s198/S198_ONE_CPU_NRA_TRAIN_REPORT.json")
    parent,trained,full,physical=physical_and_parent(
        parent_zlib=parent_z,parent_report=parent_j7,
        candidate_zlib=cand_z,train_report=cand_train,
        destination=out/"S199_PHYSICALLY_TRAINED_NRA_46M_FP16.safetensors.lzma")
    # The genuine original native S190 CPU J7 code is used without modifying
    # the selector, profile routing, Green Tunnel, or any protected gold.
    from calibri2.s160 import real_parallel_slots_j7 as s160
    from calibri2.s168 import novel_train_frames_generalization_cpu as s168
    from calibri2.s182 import real_cpu_j7_shadow_guard as s182
    from calibri2.s190 import real_cpu_new_train_paraphrase_probe as s190
    from calibri2.s112 import prompt_conditioned_first_token_mps as s112

    if (s190.PROBE_RECORDS is not None or s182.ALL_PRED
        or s160.cached_full_cpu_parallel is not s168.ORIGINAL_DECODER):
        raise AssertionError("S199 fresh isolated true CPU candidate process required")
    original_cpu_loader=s112.load_cpu_original
    original_s168_loader=s168.load_cpu_original
    original_core=s168.ProgressiveRouteCore
    original_base_decoder=s160.cached_full_cpu_parallel
    original_s168_decoder=s168.ORIGINAL_DECODER
    neural=[]
    grown=[]
    captured={}
    probe_questions={sha_text(q) for q,_a,_g in new_train_paraphrase_probes()}

    def physically_loaded_neural_core(*a,**kw):
        cpu=original_cpu_loader(*a,**kw)
        if neural:
            raise AssertionError("S199 one physical trained candidate CPU core only")
        area=NeuroRepresentationArea().eval()
        area.load_state_dict(
            {k.removeprefix(NRA_PREFIX):v.float() for k,v in full.items()
             if k.startswith(NRA_PREFIX)},strict=True)
        if not torch.equal(area.fp16_state()[NRA_PREFIX+"decoder.weight"],
                           full[NRA_PREFIX+"decoder.weight"]):
            raise AssertionError("S199 physical changed trained NRA decoder not loaded")
        area.install(cpu[0])
        area._active=True
        neural.append(area)
        return cpu

    def capture_real_decoder(*a,**kw):
        # This is the frozen real native decoder. No expected answers or
        # question SHAs enter routing; SHA only identifies posthoc transcript.
        response=original_base_decoder(*a,**kw)
        q=a[6] if len(a)>6 else kw.get("question")
        identity=sha_text(q)
        if identity in probe_questions:
            ids=list(response[0])
            if identity in captured and captured[identity]!=ids:
                raise AssertionError("S199 repeated actual generated IDs disagree")
            captured[identity]=ids
        return response

    s112.load_cpu_original=physically_loaded_neural_core
    s168.load_cpu_original=physically_loaded_neural_core
    s168.ProgressiveRouteCore=new_46m_core_type(full,grown)
    s160.cached_full_cpu_parallel=capture_real_decoder
    s168.ORIGINAL_DECODER=capture_real_decoder
    previous_argv=list(sys.argv)
    try:
        s190.main()
    finally:
        sys.argv=previous_argv
        s112.load_cpu_original=original_cpu_loader
        s168.load_cpu_original=original_s168_loader
        s168.ProgressiveRouteCore=original_core
        s160.cached_full_cpu_parallel=original_base_decoder
        s168.ORIGINAL_DECODER=original_s168_decoder
        for area in neural:
            area._active=False

    if (len(grown)!=1 or len(neural)!=1 or grown[0].rank!=1344
        or neural[0].forward_calls<=0
        or neural[0].last_coverage!=tuple(range(12))
        or not torch.equal(grown[0].export_active_fp16()[WEIGHTS_KEY],full[WEIGHTS_KEY])
        or set(captured)!=probe_questions):
        raise AssertionError("S199 real trained 46M+NRA CPU J7 model not fully executed")
    out190=json.loads((out/"S190_REAL_CPU_32_NEW_TRAIN_PARAPHRASE_PROBES.json").read_text())
    if (out190.get("original_real_CPU_FULL_EOS_456")!=456
        or out190.get("new128_real_CPU_FULL_EOS")!=128):
        raise AssertionError("S199 original 584 full/EOS protected outputs forgotten")
    if set(s182.ALL_PRED)!={"old","new","new_expected"}:
        raise AssertionError("S199 no complete 456+128 original actual CPU J7 predictions")
    old=s182.ALL_PRED["old"]
    new=s182.ALL_PRED["new"]
    expected128=s182.ALL_PRED["new_expected"]
    s87=json.loads(Path(args[-2]).read_text())
    if s87.get("stage")!="S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT":
        raise AssertionError("S199 protected real TRAIN target provenance missing")
    gold456={row["question_sha256"]:row["target_ids"]
             for row in s87["all_456_per_identity_results"]}
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    gold32={sha_text(q):tok.encode(answer,add_special_tokens=False)+[
        tok.eos_token_id] for q,answer,_group in new_train_paraphrase_probes()}
    evidence=make_real_584_certificate(
        old_pred=old,old_target=gold456,new_pred=new,new_target=expected128,
        probe_pred=captured,probe_target=gold32,eos_id=tok.eos_token_id,
        source_bpp=0.0,
        speed_ratio=float(out190["original_CPU_S101_ratio"]),
        checkpoint_sha256=physical["actual_runtime_lzma_SHA256"],
        physically_reloaded=True,frozen_parent=True,captured_before_gold=True)

    sources=[
        Path("research/calibri2/s198/one_step_real_nra.py"),
        Path("research/calibri2/s198/test_one_step_real_nra.py"),
        Path("research/calibri2/s199/real_trained_nra_cpu_j7.py"),
        Path("research/calibri2/s199/test_real_trained_nra_cpu_j7.py"),
    ]
    report={
        "stage":S199_STAGE,
        "source_S197_pinned_GitHub_run":S197_PINNED_BPP,
        "source_S198_REAL_TRAIN_GPU_or_CPU_run":S198_PROOF,
        "NRA_real_trained_gradient_steps":trained["actual_CPU_train_steps"],
        "NRA_real_trained_FP16_updates":trained["train_NRA_Fp16_changed_elements"],
        "physical_checkpoint":physical,
        "full_456_correct_EOS":evidence.old_full_eos_correct,
        "full_128_correct_EOS":evidence.new_full_eos_correct,
        "32_NEW_TRAIN_first_correct":evidence.train32_first_correct,
        "32_NEW_TRAIN_full_EOS_correct":evidence.train32_full_eos_correct,
        "new32_complete_gain_vs_S197":evidence.train32_full_eos_correct,
        "12_NRA_layers_executed":list(neural[0].last_coverage),
        "actual_neural_forward_calls":neural[0].forward_calls,
        "same_CPU_original_S101_relative_ratio":evidence.same_CPU_speed_vs_S187,
        "actual_source_S190_first_correct":out190["NEW_TRAIN_first_correct"],
        "candidate_NRA_gradients_physically_reloaded":True,
        "all_old_S142_weights_preserved_byte_exact":True,
        "no_QA_TEST_and_no_LIVE":True,
        "model_accepted":False,
    }
    size=0
    for _ in range(40):
        report["total_physical_budget"]=physical_invoice(
            parent_report=parent,physical=physical,manifest_bytes=size,source_paths=sources)
        payload=json.dumps(report,sort_keys=True,indent=2).encode()
        if len(payload)==size:break
        size=len(payload)
    else:
        raise AssertionError("S199 physical source manifest byte count did not converge")
    budget=report["total_physical_budget"]
    gate={
        "all_584_previous_answers_retain_full_EOS":
            evidence.old_full_eos_correct==456 and evidence.new_full_eos_correct==128,
        "new_full_answer_after_ONE_learner_step":
            evidence.train32_full_eos_correct>=1,
        "full_physical_bpp_architecture_under_07":budget["architectural_bpp"]<=MAX_BPP,
        "conservative_bpp_no_free_zeros_under_07":
            budget["conservative_bpp_excluding_zero_structural_growth"]<=MAX_BPP,
        "CPU_ratio_ge_09":evidence.same_CPU_speed_vs_S187>=MIN_CPU_RATIO,
    }
    report["J7_acceptance_guard"]=gate
    # This remains a research candidate until an independent heldout QA and
    # paired candidate-vs-parent CPU speed audit; never promote/deploy.
    report["research_candidate_S199_j7_gate_pass"]=all(gate.values())
    report["model_accepted"]=False
    (out/"S199_TRAINED_NRA_REAL_616_CPU_J7_BPP.json").write_text(
        json.dumps(report,sort_keys=True,indent=2))
    print(json.dumps({
        "event":S199_STAGE,
        "old_456":evidence.old_full_eos_correct,
        "new_128":evidence.new_full_eos_correct,
        "new32_full":evidence.train32_full_eos_correct,
        "new32_first":evidence.train32_first_correct,
        "bpp":budget,
        "CPU_ratio":evidence.same_CPU_speed_vs_S187,
        "J7_research_candidate_gate":gate,
        "model_accepted":False},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
